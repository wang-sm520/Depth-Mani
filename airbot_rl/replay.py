"""Replay every recorded episode in sim (joint states set directly, no physics) and write the wrist-depth BC data.

Per frame: the recorded state/action, the sim wrist depth (airbot_rl.depth encoding) and two head inputs:
  mixed/  head = the recording's own DA2 depth   (real head, sim wrist)
  sim/    head = DA2 on the sim head render       (all sim)
The can stands at its FK start position and, from the grasp frame on, rides rigidly with link6.
Each directory holds episode_XXXXXX.h5 (state, action, depth float16 [T,2,2,128,128]) and manifest.json.
"""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--real-data", default="data/airbot-can100-da2-20260929")
parser.add_argument("--output", required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--episodes", type=int, nargs="+", help="subset, for previews")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402

from airbot_depth.depth import DepthAnythingTransform  # noqa: E402
from airbot_rl import depth as wrist_encoding  # noqa: E402
from airbot_rl.assets import CAN_POSITIONS  # noqa: E402
from airbot_rl.depth import encode_wrist_depth  # noqa: E402
from airbot_rl.policy import head_depth  # noqa: E402
from airbot_rl.scene import ARM_LINKS, CAN_HEIGHT, SCALE, CanSceneCfg, head_image, wrist_depth  # noqa: E402


class Replay:
    def __init__(self, num_envs):
        cfg = CanSceneCfg(num_envs=num_envs)
        for link in ARM_LINKS:
            setattr(cfg, f"contact_{link}", None)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 200))
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.arm, self.can = self.scene["arm"], self.scene["can"]
        self.arm_ids = self.arm.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
        self.finger_ids = self.arm.find_joints(["endleft", "endright"], preserve_order=True)[0]
        self.link6 = self.arm.find_bodies("link6")[0][0]

    def pose(self, states):
        """Write [<=N,7] native states (padded with the last) and return link6 world poses."""
        n = self.scene.num_envs
        state = torch.as_tensor(np.concatenate((states, np.repeat(states[-1:], n - len(states), 0))),
                                device=self.sim.device)
        joints = self.arm.data.default_joint_pos.clone()
        joints[:, self.arm_ids] = state[:, :6]
        joints[:, self.finger_ids] = state[:, 6:] * torch.tensor([0.5, -0.5], device=self.sim.device)
        self.arm.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        self.sim.forward()
        return self.arm.data.body_pos_w[:, self.link6], self.arm.data.body_quat_w[:, self.link6]

    def episode(self, states, start, grasp_frame):
        """Yield (head uint8 RGB, wrist metres) for batches of frames."""
        n = self.scene.num_envs
        standing = self.can.data.default_root_state[:, :7].clone()
        standing[:, :3] = self.scene.env_origins + torch.tensor([*start, CAN_HEIGHT / 2], device=self.sim.device)
        standing[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim.device)
        link_pos, link_quat = self.pose(states[grasp_frame:grasp_frame + 1])
        rel_pos, rel_quat = subtract_frame_transforms(link_pos, link_quat, standing[:, :3], standing[:, 3:])
        for first in range(0, len(states), n):
            frames = np.arange(first, min(first + n, len(states)))
            link_pos, link_quat = self.pose(states[frames])
            held = torch.zeros(n, dtype=torch.bool, device=self.sim.device)
            held[:len(frames)] = torch.as_tensor(frames > grasp_frame, device=self.sim.device)
            carried = torch.cat(combine_frame_transforms(link_pos, link_quat, rel_pos, rel_quat), 1)
            self.can.write_root_pose_to_sim(torch.where(held[:, None], carried, standing))
            self.sim.forward()
            self.sim.render()
            for name in ("head_cam", "wrist_cam"):
                self.scene[name].update(0.0, force_recompute=True)
            yield (head_image(self.scene["head_cam"].data.output["rgb"][:len(frames)]),
                   wrist_depth(self.scene["wrist_cam"].data.output["distance_to_image_plane"][:len(frames)]))


def main():
    real = Path(args.real_data)
    manifest = json.loads((real / "manifest.json").read_text())
    transform = DepthAnythingTransform(manifest["depth_config"], device="cuda", local_files_only=True)
    if transform.provenance() != manifest["depth_provenance"]:
        raise ValueError("Runtime Depth Anything differs from the recording's conversion")
    starts = {p["episode"]: p for p in json.loads(CAN_POSITIONS.read_text())}
    episodes = [e for e in manifest["episodes"] if e["episode_index"] in starts
                and (args.episodes is None or e["episode_index"] in args.episodes)]
    out = Path(args.output)
    for kind in ("mixed", "sim"):
        (out / kind / "episodes").mkdir(parents=True, exist_ok=True)
    replay = Replay(args.num_envs)
    for entry in episodes:
        p = starts[entry["episode_index"]]
        with h5py.File(real / entry["path"]) as f:
            state, action, recorded = f["state"][:], f["action"][:], f["depth"][:, 0]
        head, wrist = [], []
        for rgb, metres in replay.episode(state, (p["x"], p["y"]), p["grasp_frame"]):
            head.append(head_depth(transform, rgb))
            wrist.append(encode_wrist_depth(metres).cpu())
        head, wrist = torch.cat(head).numpy(), torch.cat(wrist).numpy()
        for kind, head_view in (("mixed", recorded), ("sim", head)):
            with h5py.File(out / kind / entry["path"], "w") as f:
                f["state"], f["action"] = state, action
                f.create_dataset("depth", data=np.stack((head_view, wrist), 1).astype(np.float16), compression="lzf")
        print(json.dumps({"episode": entry["episode_index"], "frames": len(state)}), flush=True)
    for kind in ("mixed", "sim"):
        (out / kind / "manifest.json").write_text(json.dumps({
            "head": "recorded DA2" if kind == "mixed" else "DA2 on sim render at 1/%d resolution" % SCALE,
            "wrist": {"encoding": "metric", "near": wrist_encoding.NEAR, "far": wrist_encoding.FAR,
                      "source": "sim distance_to_image_plane at 1/%d resolution" % SCALE},
            "prompt": manifest["prompt"], "real_data": str(real.resolve()),
            "episodes": [{k: e[k] for k in ("episode_index", "split", "frames", "path", "instruction")}
                         for e in episodes]}, indent=1))


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
