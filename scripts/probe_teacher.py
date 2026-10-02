import argparse
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--platform", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default="reports/teacher-cpu-probe.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from depth_policy.common import atomic_json, load_config, sha256_file, source_identity

    config = load_config(args.config)
    sys.path.insert(0, str(Path(config["openpi_root"]) / "src"))
    import jax
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from openpi_client import image_tools

    expected = "cpu" if args.platform == "cpu" else "gpu"
    if os.environ.get("JAX_PLATFORMS") != args.platform:
        raise RuntimeError("Explicit JAX backend does not match requested platform")
    if args.platform == "cuda" and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        raise RuntimeError("GPU probe requires an explicit memory budget")
    if any(device.platform != expected for device in jax.devices()):
        raise RuntimeError("Unexpected JAX backend")
    if args.repeats < 2:
        raise ValueError("Need compile-inclusive and warm inference measurements")
    logging.basicConfig(level=logging.INFO)
    with h5py.File(args.episode, "r") as stream:
        metadata = json.loads(stream.attrs["metadata"])
        observation = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(
                stream["steps/rgb_agentview"][args.frame], 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(
                stream["steps/rgb_wrist"][args.frame], 224, 224)),
            "observation/state": stream["steps/state"][args.frame],
            "prompt": metadata["instruction"],
        }
    record = {
        "status": "loading", "backend": args.platform, "devices": [str(device) for device in jax.devices()],
        "checkpoint": config["teacher_checkpoint"], "policy_config": "pi05_libero",
        "episode": args.episode, "episode_sha256": sha256_file(args.episode), "frame": args.frame,
        "instruction": observation["prompt"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "openpi_source": source_identity(config["openpi_root"]), "seed": 0,
        "task_success_measured": False,
    }
    atomic_json(args.report, record)
    start = time.perf_counter()
    policy = policy_config.create_trained_policy(training_config.get_config("pi05_libero"), config["teacher_checkpoint"])
    record.update(status="loaded", load_seconds=time.perf_counter() - start, inferences=[])
    atomic_json(args.report, record)
    print(json.dumps({"loaded_seconds": record["load_seconds"]}), flush=True)
    for repeat in range(args.repeats):
        policy._rng = jax.random.key(0)
        start = time.perf_counter()
        result = policy.infer(observation)
        actions = np.asarray(result["actions"])
        duration = time.perf_counter() - start
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < config["teacher_replan_steps"]:
            raise ValueError(f"Unexpected action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite teacher action")
        np.save(Path(args.report).with_suffix(f".actions-{repeat}.npy"), actions)
        measurement = {"repeat": repeat, "seconds": duration, "action_shape": list(actions.shape),
                       "first_action": actions[0].tolist(), "min": actions.min(axis=0).tolist(),
                       "max": actions.max(axis=0).tolist()}
        record["inferences"].append(measurement)
        record["peak_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        record["status"] = "infer_complete" if repeat + 1 == args.repeats else "inferring"
        atomic_json(args.report, record)
        print(json.dumps(measurement), flush=True)
    first = np.load(Path(args.report).with_suffix(".actions-0.npy"))
    second = np.load(Path(args.report).with_suffix(".actions-1.npy"))
    record["same_seed_max_action_difference"] = float(np.max(np.abs(first - second)))
    np.testing.assert_allclose(first, second, atol=1e-5, rtol=1e-5)
    record["passed"] = True
    atomic_json(args.report, record)


if __name__ == "__main__":
    main()
