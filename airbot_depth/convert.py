"""Convert LeRobot RGB with the same frozen transform used by online inference."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.depth import DEFAULT_DEPTH_CONFIG, DepthAnythingTransform
from airbot_depth.lerobot import DEFAULT_CAMERAS, LeRobotSource


def episode_split(indices, validation_fraction, seed):
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("Select at least two unique, complete episodes")
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    count = max(1, min(len(indices) - 1, round(len(indices) * validation_fraction)))
    validation = set(np.random.default_rng(seed).permutation(sorted(indices))[:count].tolist())
    return {index: ("validation" if index in validation else "train") for index in sorted(indices)}


def verify_episode(path, source, index, manifest):
    expected = source.load_episode(index)
    with h5py.File(path, "r") as stream:
        if not stream.attrs.get("complete", False):
            raise ValueError(f"Episode write is incomplete: {path}")
        if stream.attrs.get("conversion_signature") != manifest["conversion_signature"]:
            raise ValueError("Episode belongs to a different conversion")
        if stream.attrs["episode_index"] != index or stream.attrs["instruction"] != source.prompt:
            raise ValueError("Converted episode identity mismatch")
        for key, array in expected.items():
            if not np.array_equal(stream[key][:], array):
                raise ValueError(f"Conversion changed native {key}: episode {index}")
        size = manifest["depth_config"]["image_size"]
        shape = (len(expected["state"]), len(source.camera_keys), 2, size, size)
        if stream["depth"].shape != shape or stream["depth"].dtype != np.float32:
            raise ValueError("Depth shape/dtype mismatch")
        for start in range(0, shape[0], 32):
            depth = stream["depth"][start:start + 32]
            if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
                raise ValueError("Converted depth is not finite in [0,1]")
            if not np.isin(depth[:, :, 1], [0.0, 1.0]).all():
                raise ValueError("Invalid depth mask")
            if np.any(depth[:, :, 0][depth[:, :, 1] == 0] != 0):
                raise ValueError("Invalid/padded depth pixels must be zero")
        for camera, key in enumerate(source.camera_keys):
            offset = source.episodes[index][f"videos/{key}/from_timestamp"]
            if not np.allclose(stream["video_pts"][:, camera], offset + expected["timestamp"],
                               atol=0.002, rtol=0):
                raise ValueError("Video PTS does not match state/action timestamps")
    return expected


def write_episode(destination, source, index, transform, manifest):
    values = source.load_episode(index)
    length = len(values["state"])
    size = transform.config["image_size"]
    temporary = destination.with_suffix(".partial.h5")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only the unpublished staging file for this exact episode is replaced.
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    with h5py.File(temporary, "x") as stream:
        stream.attrs["episode_index"] = index
        stream.attrs["instruction"] = source.prompt
        stream.attrs["conversion_signature"] = manifest["conversion_signature"]
        stream.attrs["complete"] = False
        for key, value in values.items():
            stream.create_dataset(key, data=value)
        depth = stream.create_dataset(
            "depth", shape=(length, len(source.camera_keys), 2, size, size), dtype="float32",
            chunks=(1, 1, 2, size, size), compression="lzf",
        )
        pts = stream.create_dataset("video_pts", shape=(length, len(source.camera_keys)), dtype="float64")
        iterators = [source.rgb_frames(index, key, values["timestamp"]) for key in source.camera_keys]
        count = 0
        try:
            for frame, views in enumerate(zip(*iterators, strict=True)):
                depth[frame] = transform([view[0] for view in views])
                pts[frame] = [view[1] for view in views]
                count += 1
        finally:
            for iterator in iterators:
                iterator.close()
        if count != length:
            raise ValueError("Decoder did not return all episode frames")
        stream.attrs["complete"] = True
        stream.flush()
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(temporary, destination)
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth-config")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = LeRobotSource(args.source, args.cameras)
    indices = sorted(args.episodes if args.episodes is not None else source.episodes)
    if any(index not in source.episodes for index in indices):
        raise ValueError("Unknown source episode index")
    splits = episode_split(indices, args.validation_fraction, args.split_seed)
    output = Path(args.output).resolve()
    if output == source.root or source.root.is_relative_to(output) or output.is_relative_to(source.root):
        raise ValueError("Conversion output must be separate from the source dataset")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty output; choose a new path or explicit --resume")
    if args.resume and not (output / "manifest.json").is_file():
        raise FileNotFoundError("Resume requires an existing conversion manifest")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".conversion.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another converter owns this output directory") from error
        run_conversion(args, source, indices, splits, output)


def run_conversion(args, source, indices, splits, output):
    config = json.loads(Path(args.depth_config).read_text()) if args.depth_config else DEFAULT_DEPTH_CONFIG
    transform = DepthAnythingTransform(config, device=args.device, local_files_only=args.local_files_only)
    code_root = Path(__file__).parent
    identity = {
        "schema_version": 1, "source": source.fingerprint(indices),
        "robot_type": source.info["robot_type"], "fps": source.fps,
        "camera_keys": source.camera_keys, "state_dim": 7, "action_dim": 7,
        "state_names": source.info["features"]["observation.state"]["names"],
        "action_names": source.info["features"]["action"]["names"],
        "state_units": ["rad"] * 6 + ["m"], "action_units": ["rad"] * 6 + ["m"],
        "action_semantics": "absolute_joint_position", "prompt": source.prompt,
        "depth_config": transform.config, "depth_provenance": transform.provenance(),
        "model_input_quantization": "float32_to_float16_to_float32",
        "split_seed": args.split_seed, "validation_fraction": args.validation_fraction,
        "episode_splits": {str(index): split for index, split in splits.items()},
        "source_code": {name: sha256_file(code_root / name)
                        for name in ("common.py", "lerobot.py", "depth.py", "convert.py")},
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if (manifest["conversion_signature"] != signature
                or any(manifest.get(key) != value for key, value in identity.items())):
            raise ValueError("Resume source/config/code/depth provenance differs; use a new output")
        recorded_ids = [entry["episode_index"] for entry in manifest["episodes"]]
        if len(set(recorded_ids)) != len(recorded_ids) or not set(recorded_ids).issubset(indices):
            raise ValueError("Resume manifest has duplicate or unknown episodes")
        for entry in manifest["episodes"]:
            index = entry["episode_index"]
            if (entry["split"] != splits[index] or entry["instruction"] != source.prompt
                    or entry["frames"] != source.episodes[index]["length"]
                    or entry["path"] != f"episodes/episode_{index:06d}.h5"):
                raise ValueError("Resume episode metadata differs from the frozen source/split")
            if sha256_file(output / entry["path"]) != entry["sha256"]:
                raise ValueError("Previously converted episode was modified")
    else:
        manifest = {**identity, "conversion_signature": signature,
                    "status": "converting", "episodes": [],
                    "started_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(output / "manifest.json", manifest)
    done = {entry["episode_index"] for entry in manifest["episodes"]}
    for index in indices:
        if index in done:
            continue
        path = output / "episodes" / f"episode_{index:06d}.h5"
        elapsed = 0.0
        if not path.exists():
            elapsed = write_episode(path, source, index, transform, manifest)
        values = verify_episode(path, source, index, manifest)
        entry = {"episode_index": index, "split": splits[index],
                 "instruction": source.prompt, "frames": len(values["state"]),
                 "path": str(path.relative_to(output)), "sha256": sha256_file(path),
                 "conversion_seconds": elapsed}
        manifest["episodes"].append(entry)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "manifest.json", manifest)
        print(json.dumps({"episode": index, "split": splits[index], "frames": entry["frames"],
                          "completed_episodes": len(manifest["episodes"]), "seconds": elapsed}), flush=True)
    manifest["episodes"].sort(key=lambda entry: entry["episode_index"])
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_frames"] = sum(entry["frames"] for entry in manifest["episodes"])
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "episodes": len(manifest["episodes"]),
                      "frames": manifest["total_frames"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
