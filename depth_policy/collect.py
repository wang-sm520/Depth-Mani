import argparse
import collections
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from depth_policy.common import ROOT, load_config, sha256_file, source_identity, validate_action
from depth_policy.episodes import EpisodeWriter, episode_index
from depth_policy.manifest import load_manifest
from depth_policy.simulation import CAMERAS, camera_metadata, make_env, observe, task_and_states


class Teacher:
    def __init__(self, config):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(
            "127.0.0.1", config["teacher_port"], connect_timeout_s=15, inference_timeout_s=180
        )
        self.metadata = self.client.get_server_metadata()
        if self.metadata.get("policy_config") != "pi05_libero":
            raise ValueError(f"Wrong teacher: {self.metadata}")
        expected = Path(config["teacher_checkpoint"]).resolve()
        if Path(self.metadata.get("checkpoint_path", "")).resolve() != expected:
            raise ValueError("Teacher checkpoint path differs from configuration")
        if self.metadata.get("request_seed_protocol") != 1:
            raise ValueError("Use scripts/serve_teacher.py for reproducible per-request seeds")

    def reset(self, seed):
        self.generator = np.random.default_rng(seed)

    def infer(self, observation, instruction):
        from openpi_client import image_tools

        output = self.client.infer({
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation["rgb_agentview"], 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation["rgb_wrist"], 224, 224)),
            "observation/state": observation["state"], "prompt": instruction,
            "depth_policy_request_seed": int(self.generator.integers(0, 2**32)),
        })
        actions = np.asarray(output["actions"])
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("Malformed teacher action chunk")
        return actions


class DummyPolicy:
    metadata = {"policy_config": "dummy_smoke_motion_v2"}

    def infer(self, observation, instruction):
        return np.tile([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -1.0], (8, 1))


def run_episode(environment, task, initial_state, metadata, config, policy, directory, stop):
    import robosuite

    np.random.seed(metadata["seed"])
    environment.seed(metadata["seed"])
    environment.reset()
    raw = environment.set_init_state(initial_state)
    if hasattr(policy, "reset"):
        policy.reset(metadata["seed"])
    metadata = {**metadata, "instruction": task.language, "camera": camera_metadata(environment),
                "controller_config": robosuite.load_controller_config(default_controller="OSC_POSE")}
    writer = EpisodeWriter(directory, metadata, environment.get_sim_state(), environment.sim.model.get_xml())
    queue = collections.deque()
    timings = []
    success, reason, error = False, "timeout", None
    try:
        for index in range(config["settle_steps"] + config["max_steps"]):
            if stop["requested"]:
                reason = "interrupted"
                break
            observation = observe(environment, raw)
            phase = int(index >= config["settle_steps"])
            if not phase:
                action = validate_action([0.0] * 6 + [-1.0])
            else:
                if not queue:
                    start = time.perf_counter()
                    actions = policy.infer(observation, task.language)
                    timings.append(time.perf_counter() - start)
                    if len(actions) < config["teacher_replan_steps"]:
                        raise ValueError("Action chunk shorter than replan interval")
                    queue.extend(actions[:config["teacher_replan_steps"]])
                action = validate_action(queue.popleft())
            poses = np.stack([
                np.concatenate([environment.sim.data.cam_xpos[environment.sim.model.camera_name2id(camera)],
                                environment.sim.data.cam_xmat[environment.sim.model.camera_name2id(camera)]])
                for camera in CAMERAS
            ])
            timestamp = environment.sim.data.time
            sim_state = environment.get_sim_state().copy()
            raw, _, done, _ = environment.step(action.tolist())
            writer.append(observation, action, index, timestamp, sim_state, phase, poses)
            success = bool(environment.check_success())
            if success:
                reason = "success"
                break
            if done:
                reason = "environment_done_without_success"
                break
    except BaseException as exception:
        error = traceback.format_exc()
        reason = "interrupted" if isinstance(exception, KeyboardInterrupt) else "exception"
        success = False
    destination = writer.finish(success, reason, environment.get_sim_state(), timings, error)
    result = {"path": str(destination), "success": success, "reason": reason,
              "steps": writer.count, "initial_state_id": metadata["initial_state_id"]}
    print(json.dumps(result), flush=True)
    if error:
        raise RuntimeError(f"Episode failed; saved to {destination}\n{error}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--target-successes", type=int)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("evaluation_only"):
        raise ValueError("Evaluation-only configuration cannot collect training demonstrations")
    if args.max_steps:
        config["max_steps"] = args.max_steps
    _, states, initial_file = task_and_states(config)
    manifest = load_manifest(config, states, initial_file)
    candidates = [entry for entry in manifest["states"] if entry["split"] == args.split]
    directory = args.output or ("data/smoke" if args.dummy else config["data_dir"])
    Path(directory).mkdir(parents=True, exist_ok=True)
    lock = (Path(directory) / ".collector.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = episode_index(directory)
    base_seed = config["seed"]
    policy = DummyPolicy() if args.dummy else Teacher(config)
    identity_keys = ("policy_config", "checkpoint_path", "request_seed_protocol")
    if any(any(item["teacher"].get(key) != policy.metadata.get(key) for key in identity_keys)
           for item in previous):
        raise ValueError("Output directory already contains another teacher's episodes")
    environment, task = make_env(config, base_seed)
    provenance = {
        "task_name": config["task"], "suite": config["suite"], "split": args.split,
        "teacher": policy.metadata, "depth_units": "m", "orientation": "flip_both_axes",
        "control_freq": config["control_freq"], "resolution": config["resolution"],
        "runtime_environment": {name: os.environ.get(name) for name in [
            "MUJOCO_GL", "PYOPENGL_PLATFORM", "NUMBA_DISABLE_JIT", "CUDA_VISIBLE_DEVICES"]},
        "state_convention": "eef_xyz_m + eef_axisangle_rad + two_gripper_joint_positions_m",
        "action_convention": "OSC_POSE input: delta_xyz + delta_axisangle + gripper; controller scales inputs",
        "config": config, "manifest_sha256": sha256_file(config["manifest"]),
        "collector_source": source_identity(ROOT / "depth_policy"),
        "openpi_source": source_identity(config["openpi_root"]),
        "versions": {name: importlib.metadata.version(name)
                     for name in ["numpy", "torch", "robosuite", "mujoco", "h5py"]},
    }
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for attempt in range(args.attempts):
            matching = [item for item in previous if item["split"] == args.split]
            if args.target_successes and sum(item["success"] for item in matching) >= args.target_successes:
                break
            if stop["requested"]:
                break
            counts = collections.Counter(item["initial_state_id"] for item in matching)
            entry = min(candidates, key=lambda item: (counts[item["id"]], item["id"]))
            metadata = {**provenance, "initial_state_id": entry["id"],
                        "initial_state_sha256": entry["sha256"],
                        "seed": base_seed + entry["id"] + counts[entry["id"]] * len(states)}
            result = run_episode(environment, task, states[entry["id"]], metadata,
                                 config, policy, directory, stop)
            previous.append({**metadata, **result})
    finally:
        environment.close()
        lock.close()


if __name__ == "__main__":
    main()
