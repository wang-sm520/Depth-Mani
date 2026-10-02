"""Residual PPO env: the frozen depth BC proposes the next 4 targets and PPO adds a bounded residual.

One env step is one policy decision, exactly as deployed: observe, predict an 8x7 chunk, execute its
first 4 targets, each expanded into 25 Hz waypoints by the deployment client's interpolation, reaching
the drives COMMAND_DELAY later as on the real arm.
"""

import json
import math

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from airbot_rl.policy import FrozenBC
from airbot_rl.assets import CAN_POSITIONS
from airbot_rl.scene import ARM_LINKS, CAN_HEIGHT, CAN_RADIUS, CONTROL_HZ, PHYSICS_HZ, SUBSTEPS, CanSceneCfg, head_image, wrist_depth

EXECUTE = 4  # targets run per decision (deployment --chunk-size-execute)
# Deployment client (airbot_deploy/client.py) limits and interpolation increments; sim and robot share them.
LOWER = (-3.14, -2.96, -0.087, -3.01, -1.76, -3.01, 0.0)
UPPER = (2.09, 0.17, 3.14, 3.01, 1.76, 3.01, 0.072)
STEP_LENGTH = (0.01,) * 6 + (0.005,)
COMMAND_DELAY = 10  # physics steps (100 ms) from target to drive, identified with scene.py's arm gains
HOME_EEF = 0.01  # recorded episodes start with the gripper closed at 0.007-0.01 m
CAN_JITTER = 0.01  # m, around each recorded start position
PAD_CENTER = (0.0, 0.0, 0.005)  # finger pad centre in each finger body; their midpoint is the grasp point


@configclass
class CanEnvCfg(DirectRLEnvCfg):
    checkpoint = "airbot_rl/runs/bc_wrist_v1/step_006000.pt"

    # Gates and rewards are user-confirmed 10-03.
    max_targets = 1000  # user-confirmed 10-03
    max_joint_step = 0.1  # user-confirmed 10-03
    success_radius = 0.05  # user-confirmed 10-03
    success_can_bottom = 0.25  # user-confirmed 10-03
    residual_scale = (0.05,) * 6 + (0.01,)  # user-confirmed 10-03
    success_reward = 10.0  # user-confirmed 10-03
    limit_penalty = 100.0  # user-confirmed 10-03
    contact_penalty = 1.0  # user-confirmed 10-03
    topple_penalty = 0.5  # user-confirmed 10-03
    topple_angle = math.radians(45.0)  # user-confirmed 10-03

    decimation = SUBSTEPS  # replaced every step by SUBSTEPS x the number of waypoints to run
    episode_length_s = max_targets / CONTROL_HZ  # nominal; max_episode_length counts decisions instead
    # Cameras are rendered explicitly at decisions. A huge render_interval stalls Isaac's reset (1e5 did,
    # 1000 starts fine), so keep 1000: one stray in-loop render per 5 s of sim time.
    sim = SimulationCfg(dt=1 / PHYSICS_HZ, render_interval=1000,
                        render=RenderCfg(antialiasing_mode="FXAA"))  # no temporal AA across sparse frames
    scene = CanSceneCfg(num_envs=8)
    action_space = EXECUTE * 7
    observation_space = 512 + 7 + EXECUTE * 7
    state_space = 26 + EXECUTE * 7


class CanEnv(DirectRLEnv):
    cfg: CanEnvCfg

    def __init__(self, cfg: CanEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.arm, self.can = self.scene["arm"], self.scene["can"]
        self.contacts = [self.scene[f"contact_{link}"] for link in ARM_LINKS]
        self.arm_ids = self.arm.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        self.finger_ids = self.arm.find_joints(["endleft", "endright"], preserve_order=True)[0]
        self.pad_ids = self.arm.find_bodies(["left", "right"], preserve_order=True)[0]
        tensor = lambda values: torch.tensor(values, device=self.device)  # noqa: E731
        self.lower, self.upper, self.step_length = tensor(LOWER), tensor(UPPER), tensor(STEP_LENGTH)
        self.residual_scale = tensor(cfg.residual_scale)
        self.bc = FrozenBC(cfg.checkpoint, str(self.device))
        # Can starts: one FK grasp midpoint per retained recording, in the env frame.
        starts = json.loads(CAN_POSITIONS.read_text())
        self.can_starts = torch.tensor([[p["x"], p["y"]] for p in starts], device=self.device)
        n = self.num_envs
        self.command = torch.zeros(n, 7, device=self.device)  # last sent target, as deployment's previous_action
        self.bc_chunk = torch.zeros(n, EXECUTE, 7, device=self.device)
        self.home = torch.zeros(n, 3, device=self.device)
        self.excess = torch.zeros(n, device=self.device)
        self.touched = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.touched_dog = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.touched_floor = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.toppled = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.episode_toppled = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.queue = torch.zeros(COMMAND_DELAY, n, 7, device=self.device)  # targets in flight, kept across decisions
        self.tick = 0
        self.current_ticks = 0

    @property
    def max_episode_length(self):
        return self.cfg.max_targets // EXECUTE

    # --- helpers -------------------------------------------------------------------------------------------
    def native_state(self):
        q = self.arm.data.joint_pos
        return torch.cat((q[:, self.arm_ids], q[:, self.finger_ids[:1]] - q[:, self.finger_ids[1:]]), dim=1)

    def native_to_sim(self, native):
        target = self.arm.data.joint_pos_target.clone()
        target[:, self.arm_ids] = native[:, :6]
        target[:, self.finger_ids] = native[:, 6:] * torch.tensor([0.5, -0.5], device=self.device)
        return target

    def grasp_point(self):
        pads = self.arm.data.body_pos_w[:, self.pad_ids] + quat_apply(
            self.arm.data.body_quat_w[:, self.pad_ids], torch.tensor(PAD_CENTER, device=self.device).expand(
                self.num_envs, 2, 3))
        return pads.mean(1) - self.scene.env_origins

    def can_tilt_cos(self):
        q = self.can.data.root_quat_w
        return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # z-component of the can axis

    def camera_inputs(self):
        """(head uint8 RGB, wrist metres), both on the real cameras' pixel grids at 1/SCALE."""
        return (head_image(self.scene["head_cam"].data.output["rgb"]),
                wrist_depth(self.scene["wrist_cam"].data.output["distance_to_image_plane"]))

    # --- DirectRLEnv hooks ----------------------------------------------------------------------------------
    def _pre_physics_step(self, actions):
        raw = self.bc_chunk + self.residual_scale * torch.tanh(actions.view(-1, EXECUTE, 7))
        targets = torch.clamp(raw, self.lower, self.upper)
        self.excess = (raw - targets).abs().sum((1, 2))
        starts, ends, previous = [], [], self.command
        for target in targets.unbind(1):
            step = target - previous
            capped = torch.cat((step[:, :6].clamp(-self.cfg.max_joint_step, self.cfg.max_joint_step), step[:, 6:]), 1)
            self.excess += (step - capped).abs().sum(1)
            starts.append(previous)
            previous = previous + capped
            ends.append(previous)
        self.command = previous
        starts, ends = torch.stack(starts, 1), torch.stack(ends, 1)
        # Deployment: ceil(max |delta| / STEP_LENGTH) waypoints per target, linspace(previous, target)[1:].
        counts = ((ends - starts).abs() / self.step_length).ceil().amax(-1).clamp_min(1)
        done = counts.cumsum(1)
        ticks = int(done[:, -1].max())
        tick = torch.arange(1, ticks + 1, device=self.device).view(1, -1, 1)
        segment = (tick > done[:, None]).sum(-1).clamp(max=EXECUTE - 1)  # finished envs hold their last target
        fraction = ((tick[..., 0] - (done - counts).gather(1, segment)) / counts.gather(1, segment)).clamp(max=1)
        index = segment[..., None].expand(-1, -1, 7)
        start = starts.gather(1, index)
        self.waypoints = start + (ends.gather(1, index) - start) * fraction[..., None]
        self.cfg.decimation = SUBSTEPS * ticks
        self.current_ticks = ticks
        self.substep = 0
        self.touched[:] = False
        self.touched_dog[:] = False
        self.touched_floor[:] = False

    def _update_touched(self, length):
        for sensor in self.contacts:
            force = sensor.data.force_matrix_w_history[:, :length]
            touched = force.norm(dim=-1).amax(dim=1).gt(1.0)
            dog = touched[..., :-1].flatten(1).any(1)
            floor = touched[..., -1:].flatten(1).any(1)
            self.touched |= touched.flatten(1).any(1)
            self.touched_dog |= dog
            self.touched_floor |= floor

    def _apply_action(self):
        slot = self.tick % COMMAND_DELAY
        delayed = self.queue[slot].clone()
        self.queue[slot] = self.waypoints[:, self.substep // SUBSTEPS]
        self.tick += 1
        self.arm.set_joint_position_target(self.native_to_sim(delayed))
        if self.substep and self.substep % SUBSTEPS == 0:
            self._update_touched(SUBSTEPS)
        self.substep += 1

    def _get_observations(self):
        self.sim.render()
        state = self.native_state()
        chunk, features, norm_state = self.bc(state, *self.camera_inputs())
        self.bc_chunk = chunk[:, :EXECUTE]
        can_pos = self.can.data.root_pos_w - self.scene.env_origins
        grasp = self.grasp_point()
        privileged = torch.cat((state, self.arm.data.joint_vel[:, self.arm_ids], can_pos, self.can.data.root_quat_w,
                                can_pos - grasp, grasp - self.home), dim=1)
        return {"policy": torch.cat((features, norm_state, self.bc_chunk.flatten(1)), 1),
                "critic": torch.cat((privileged, self.bc_chunk.flatten(1)), 1)}

    def _get_dones(self):
        self._update_touched(min(self.current_ticks, SUBSTEPS))
        cos = self.can_tilt_cos().abs()
        lowest = self.can.data.root_pos_w[:, 2] - CAN_HEIGHT / 2 * cos - CAN_RADIUS * (1 - cos ** 2).sqrt()
        home = (self.grasp_point() - self.home).norm(dim=1) < self.cfg.success_radius
        self.success = home & (lowest > self.cfg.success_can_bottom)
        return self.success, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        toppled = (self.can_tilt_cos() < math.cos(self.cfg.topple_angle)) & ~self.toppled
        self.toppled |= toppled
        self.episode_toppled = self.toppled.clone()
        reward = (self.cfg.success_reward * self.success - self.cfg.limit_penalty * self.excess
                  - self.cfg.contact_penalty * self.touched - self.cfg.topple_penalty * toppled)
        logs = {"success": self.success.float().mean(), "clip_excess": self.excess.mean(),
                "contact": self.touched.float().mean(), "toppled": self.toppled.float().mean()}
        ended = self.reset_terminated | self.reset_time_outs
        if ended.any():
            logs["episode_success"] = self.success[ended].float().mean()
        self.extras["log"] = logs
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dog = self.scene["dog"]
        dog.write_joint_state_to_sim(dog.data.default_joint_pos[env_ids], dog.data.default_joint_vel[env_ids],
                                     env_ids=env_ids)
        joints = self.arm.data.default_joint_pos[env_ids].clone()
        joints[:, self.finger_ids] = torch.tensor([HOME_EEF / 2, -HOME_EEF / 2], device=self.device)
        self.arm.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        self.arm.set_joint_position_target(joints, env_ids=env_ids)
        pose = self.can.data.default_root_state[env_ids].clone()
        pose[:, :2] = self.can_starts[torch.randint(len(self.can_starts), (n,), device=self.device)]
        pose[:, :2] += sample_uniform(-CAN_JITTER, CAN_JITTER, (n, 2), self.device)
        pose[:, :3] += self.scene.env_origins[env_ids]
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.can.write_root_state_to_sim(pose, env_ids)
        self.sim.forward()  # refresh link poses before reading them and before the next render
        self.command[env_ids] = self.native_state()[env_ids]
        self.queue[:, env_ids] = self.command[env_ids]
        self.home[env_ids] = self.grasp_point()[env_ids]
        self.toppled[env_ids] = False
