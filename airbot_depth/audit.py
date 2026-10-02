"""Audit offline Airbot conversion/training and RGB-to-action inference.

These measurements use recorded observations. They are not robot rollouts and
cannot measure task success, recovery behavior, or deployment control safety.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.convert import episode_split, verify_episode
from airbot_depth.data import AirbotDataset, fit_statistics, load_manifest, select_episodes
from airbot_depth.lerobot import LeRobotSource
from airbot_depth.policy import AirbotDepthPolicy
from airbot_depth.train import environment_identity


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = (
    "schema_version", "source", "robot_type", "fps", "camera_keys", "state_dim", "action_dim",
    "state_names", "action_names", "state_units", "action_units", "action_semantics", "prompt",
    "depth_config", "depth_provenance", "model_input_quantization", "split_seed",
    "validation_fraction", "episode_splits", "source_code",
)
CONVERTER_LOCK_CHANGE = (
    "c1e45dc0a487cfe0c2947fe7cb50c2369edd729af644aaa343de056a8c3b3b25",
    "c1f3f665b9ba8e8f23a811d151d5a1cd8a2a1e31ca11ee942e9c451dcd626915",
)


def maximum_difference(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or not first.size:
        raise ValueError("Comparison requires equal, nonempty array shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Comparison contains nonfinite values")
    return float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def validate_contract(manifest, run, manifest_sha256):
    if manifest.get("status") != "complete" or run.get("status") != "complete":
        raise ValueError("Audit requires completed conversion and training")
    if run.get("manifest_sha256") != manifest_sha256 or run.get("manifest") != manifest:
        raise ValueError("Training manifest hash/content differs from the converted dataset")
    if run.get("initialization") != "all_policy_parameters_random":
        raise ValueError("Run does not declare random policy initialization")
    if manifest["state_dim"] != 7 or manifest["action_dim"] != 7:
        raise ValueError("Physical-unit audit requires six joints and one gripper")
    if (manifest["state_units"] != ["rad"] * 6 + ["m"]
            or manifest["action_units"] != ["rad"] * 6 + ["m"]
            or manifest["action_semantics"] != "absolute_joint_position"):
        raise ValueError("Unverified action/state units or semantics")
    identity = {key: manifest[key] for key in IDENTITY_FIELDS}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if signature != manifest["conversion_signature"]:
        raise ValueError("Saved conversion signature does not match its saved identity")
    indices = manifest["source"]["selected_episode_indices"]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate selected source episodes")
    expected = episode_split(indices, manifest["validation_fraction"], manifest["split_seed"])
    if {str(key): value for key, value in expected.items()} != manifest["episode_splits"]:
        raise ValueError("Episode split differs from the frozen split seed/fraction")
    episodes = manifest["episodes"]
    if sorted(item["episode_index"] for item in episodes) != sorted(indices):
        raise ValueError("Converted episode coverage differs from the selected source episodes")
    if any(item["split"] != expected[item["episode_index"]] for item in episodes):
        raise ValueError("Converted episode split differs from the frozen split")
    training, validation = [select_episodes(manifest, split) for split in ("train", "validation")]
    for key in ("episode_index", "path", "sha256"):
        if {item[key] for item in training} & {item[key] for item in validation}:
            raise ValueError(f"Train/validation leakage by {key}")
    for key, actual in (("train_episodes", len(training)), ("validation_episodes", len(validation)),
                        ("train_frames", sum(item["frames"] for item in training)),
                        ("validation_frames", sum(item["frames"] for item in validation))):
        if run[key] != actual:
            raise ValueError(f"Run count differs from complete episode coverage: {key}")
    if run["completed_steps"] != run["config"]["steps"]:
        raise ValueError("Training did not finish its configured number of steps")
    full_source = len(indices) == manifest["source"]["total_episodes"]
    formal = full_source and len(indices) == 200
    if formal and (len(training), len(validation)) != (180, 20):
        raise ValueError("The complete 200-episode experiment requires a 180/20 split")
    return {"scope": "full_200_episode_dataset" if formal else "subset_or_other_dataset",
            "full_source_selected": full_source, "formal_200_episode_dataset": formal,
            "formal_180_train_20_validation": formal and len(training) == 180 and len(validation) == 20,
            "train_episode_indices": [item["episode_index"] for item in training],
            "validation_episode_indices": [item["episode_index"] for item in validation],
            "train_episodes": len(training), "validation_episodes": len(validation),
            "train_frames": run["train_frames"], "validation_frames": run["validation_frames"],
            "whole_episode_split_verified": True, "split_overlap": [],
            "conversion_signature_verified": True, "run_manifest_verified": True}


def audit_source_code(manifest, run):
    records = []
    strict = len(manifest["source"]["selected_episode_indices"]) == manifest["source"]["total_episodes"] == 200
    for phase, saved in (("conversion", {f"airbot_depth/{key}": value
                                         for key, value in manifest["source_code"].items()}),
                         ("training", run["source"])):
        for name, previous in sorted(saved.items()):
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f"Saved source file cannot be audited: {name}")
            current = sha256_file(path)
            record = {"phase": phase, "path": name, "saved_sha256": previous,
                      "current_sha256": current, "unchanged": previous == current}
            if previous != current:
                record["note"] = "Saved and current source differ; this audit does not claim source equality."
                if name == "airbot_depth/convert.py" and (previous, current) == CONVERTER_LOCK_CHANGE:
                    record["note"] = "Known change: added exclusive output-directory conversion lock."
                if name in {"airbot_depth/depth.py", "airbot_depth/model.py", "depth_policy/model.py"}:
                    raise ValueError(f"Inference implementation changed since {phase}: {name}")
                if strict:
                    raise ValueError(f"Formal experiment source changed since {phase}: {name}")
            records.append(record)
    return {"files": records, "all_saved_files_unchanged": all(item["unchanged"] for item in records),
            "changed_files": sorted({item["path"] for item in records if not item["unchanged"]}),
            "depth_and_policy_model_unchanged": True, "formal_strict_source_equality": strict,
            "audit_source_sha256": sha256_file(__file__)}


def validate_checkpoint(payload, manifest, run, manifest_sha256):
    if payload.get("manifest_sha256") != manifest_sha256 or payload.get("manifest") != manifest:
        raise ValueError("Checkpoint uses a different manifest")
    for name in ("config", "model_config", "statistics", "vocabulary", "depth_config", "depth_provenance",
                 "source", "environment", "action_semantics", "image_input_precision"):
        if payload.get(name) != run.get(name):
            raise ValueError(f"Checkpoint differs from run metadata: {name}")
    if type(payload["step"]) is not int or not 1 <= payload["step"] <= run["completed_steps"]:
        raise ValueError("Invalid checkpoint training step")
    for key in ("validation_loss", "best_validation_loss"):
        if not np.isfinite(payload[key]):
            raise ValueError(f"Nonfinite checkpoint metric: {key}")


def action_error_summary(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 7 or not len(target):
        raise ValueError("Physical action errors require nonempty [frames,7] arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite native action prediction/target")
    error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    mae = error.mean(axis=0)
    return {"validation_frames": len(target), "joint_mean_mae_rad": float(mae[:6].mean()),
            "joint_mae_rad": mae[:6].tolist(), "gripper_mae_mm": float(mae[6] * 1000),
            "joint_max_error_rad": float(error[:, :6].max()),
            "gripper_max_error_mm": float(error[:, 6].max() * 1000)}


@torch.inference_mode()
def evaluate_first_actions(policy, dataset, native_actions, batch_size):
    predictions = []
    for start in range(0, len(dataset), batch_size):
        batch = default_collate([dataset[index] for index in range(start, min(start + batch_size, len(dataset)))])
        if not bool(batch["mask"][:, 0].all()):
            raise ValueError("Every recorded observation needs its corresponding first action")
        values = policy.model(batch["state"].to(policy.device), batch["tokens"].to(policy.device),
                              batch["images"].to(policy.device))[:, 0].cpu().numpy()
        values = values * policy.statistics["action_std"] + policy.statistics["action_mean"]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite first-action policy prediction")
        predictions.append(values)
    return action_error_summary(np.concatenate(predictions), native_actions)


def joint_limit_summary(actions, threshold=0.17):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Limit audit requires finite native seven-dimensional actions")
    joint = actions[:, 1].astype(np.float64)
    return {"frames": len(actions), "above_reference_count": int((joint > threshold).sum()),
            "raw_min_rad": float(joint.min()), "raw_max_rad": float(joint.max())}


def episode_joint_limit_summary(episode_actions):
    above = [index for index, actions in sorted(episode_actions.items())
             if np.any(np.asarray(actions)[:, 1].astype(np.float64) > 0.17)]
    return {**joint_limit_summary(np.concatenate(list(episode_actions.values()))),
            "total_episodes": len(episode_actions), "episode_count_above_reference": len(above),
            "episode_indices_above_reference": above}


def selected_roundtrip_indices(manifest, requested):
    indices = sorted(item["episode_index"] for item in manifest["episodes"])
    chosen = list(dict.fromkeys(requested if requested is not None else
                                [indices[0], indices[len(indices) // 2], indices[-1]]))
    if not chosen or any(index not in indices for index in chosen):
        raise ValueError("Roundtrip episode selection must be a nonempty subset of converted episodes")
    return chosen


def audit_roundtrip(policy, source, data_dir, manifest, indices, atol):
    if source.camera_keys != manifest["camera_keys"] or policy.camera_keys != manifest["camera_keys"]:
        raise ValueError("Roundtrip camera field order differs from the saved manifest")
    entries = {item["episode_index"]: item for item in manifest["episodes"]}
    records, examples = [], []
    for index in indices:
        entry = entries[index]
        with h5py.File(data_dir / entry["path"], "r") as stream:
            frame_indices = sorted({0, entry["frames"] // 2, entry["frames"] - 1})
            stamps = stream["timestamp"][frame_indices]
            views = {}
            for key in source.camera_keys:
                views[key] = list(source.rgb_frames(index, key, stamps))
                if len(views[key]) != len(frame_indices):
                    raise ValueError("Roundtrip RGB decoder returned the wrong sample count")
            for offset, frame in enumerate(frame_indices):
                rgb = {key: views[key][offset][0] for key in source.camera_keys}
                rgb_hashes = {key: hashlib.sha256(np.ascontiguousarray(rgb[key]).tobytes()).hexdigest()
                              for key in source.camera_keys}
                state, saved_depth = stream["state"][frame], stream["depth"][frame]
                online_action = policy.infer_rgb(state, rgb)
                transform = policy._depth_transform
                if transform.config != manifest["depth_config"] or transform.provenance() != manifest["depth_provenance"]:
                    raise ValueError("Runtime depth provenance differs from the saved conversion")
                recomputed = transform([rgb[key] for key in source.camera_keys])
                offline_action = policy.infer_depth(state, saved_depth)
                depth_difference = maximum_difference(saved_depth, recomputed)
                action_difference = maximum_difference(offline_action, online_action)
                if any(hashlib.sha256(np.ascontiguousarray(rgb[key]).tobytes()).hexdigest() != rgb_hashes[key]
                       for key in source.camera_keys):
                    raise ValueError("Inference changed the shared source RGB arrays")
                if not np.array_equal(saved_depth[:, 1], recomputed[:, 1]):
                    raise ValueError(f"Roundtrip validity mask differs: episode {index}, frame {frame}")
                if depth_difference > atol or action_difference > atol:
                    raise ValueError(f"Roundtrip mismatch: episode={index}, frame={frame}, "
                                     f"depth_maxdiff={depth_difference}, action_maxdiff={action_difference}, atol={atol}")
                records.append({"episode_index": index, "split": entry["split"], "frame_index": frame,
                                "timestamp": float(stamps[offset]),
                                "camera_video_pts": {key: float(views[key][offset][1]) for key in source.camera_keys},
                                "camera_rgb_inputs": {key: {"sha256": rgb_hashes[key], "shape": list(rgb[key].shape),
                                                            "dtype": str(rgb[key].dtype)} for key in source.camera_keys},
                                "depth_max_abs_difference": depth_difference,
                                "action_chunk_max_abs_difference": action_difference,
                                "mask_exact_match": True, "same_rgb_inputs_reused_for_both_paths": True})
                examples.append({"episode_index": index, "split": entry["split"], "frame_index": frame,
                                 "timestamp": float(stamps[offset]), "rgb": rgb, "depth": saved_depth, "state": state})
    return {"checkpoint": policy.checkpoint_path.name, "checkpoint_sha256": policy.checkpoint_sha256,
            "episode_indices": indices, "frames": records, "absolute_tolerance": atol,
            "depth_max_abs_difference": max(item["depth_max_abs_difference"] for item in records),
            "action_chunk_max_abs_difference": max(item["action_chunk_max_abs_difference"] for item in records),
            "runtime_depth_provenance_exact_match": True, "camera_field_order_verified": True,
            "camera_keys": source.camera_keys, "passed": True}, examples


def measure_latency(policy, examples, samples=20, warmup=3):
    if samples < 10 or warmup < 1:
        raise ValueError("Latency audit requires >=10 measured calls and >=1 warmup call")
    def synchronize():
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
    for index in range(warmup):
        sample = examples[index % len(examples)]
        policy.infer_rgb(sample["state"], sample["rgb"])
    before = gpu_process_snapshot(policy.device)
    durations = []
    for index in range(samples):
        sample = examples[index % len(examples)]
        synchronize()
        start = time.perf_counter()
        policy.infer_rgb(sample["state"], sample["rgb"])
        synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    if not np.isfinite(durations).all() or min(durations) <= 0:
        raise ValueError("Invalid inference timing samples")
    p50, p95 = np.percentile(durations, [50, 95])
    after = gpu_process_snapshot(policy.device)
    return {"device": str(policy.device), "camera_count": len(policy.camera_keys),
            "samples": samples, "warmup_calls": warmup, "milliseconds": durations,
            "mean_ms": float(np.mean(durations)), "p50_ms": float(p50), "p95_ms": float(p95),
            "min_ms": min(durations), "max_ms": max(durations),
            "gpu_processes_before": before, "gpu_processes_after": after,
            "load_note": "Observed latency on this host under the recorded GPU process load; "
                         "external processes can compete for GPU time. This is not an Orin benchmark.",
            "dataset_frame_period_ms": 1000.0 / policy.fps,
            "scope": "Two-camera RGB arrays through Depth Anything, preprocessing, policy and native action output; "
                     "excludes camera acquisition, video decoding, model loading, networking and robot I/O."}


def gpu_process_snapshot(device):
    if device.type != "cuda":
        return {"status": "not_applicable", "processes": []}
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        processes = []
        for row in csv.reader(result.stdout.splitlines()):
            if not row:
                continue
            pid, name, memory = [field.strip() for field in row]
            processes.append({"pid": int(pid), "process_name": name, "used_memory_mib": memory,
                              "this_audit": int(pid) == os.getpid()})
        return {"status": "captured", "processes": processes,
                "other_gpu_processes_present": any(not item["this_audit"] for item in processes)}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"status": "unavailable", "reason": str(error), "processes": []}


def load_metrics(run_dir, steps):
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    previous = 0
    for record in records:
        if type(record["step"]) is not int or not previous < record["step"] <= steps:
            raise ValueError("Training metrics have duplicate/out-of-order/invalid steps")
        if any(not np.isfinite(value) for value in record.values() if isinstance(value, (int, float))):
            raise ValueError("Nonfinite training metric")
        previous = record["step"]
    if not records or previous != steps or not any("validation_loss" in item for item in records):
        raise ValueError("Training metrics do not cover final step and held-out validation")
    return records


def write_figures(output, metrics, examples, camera_keys):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot([item["step"] for item in metrics], [item["train_loss"] for item in metrics],
            label="Training minibatch loss", alpha=0.65)
    validation = [item for item in metrics if "validation_loss" in item]
    ax.plot([item["step"] for item in validation], [item["validation_loss"] for item in validation],
            marker="o", markersize=3, label="Whole held-out validation loss")
    ax.set(xlabel="Optimizer step", ylabel="Masked Smooth-L1, normalized actions",
           title="Recorded-data prediction loss (not task success)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "loss_curve.png", dpi=150)
    plt.close(fig)

    columns = 3 * len(camera_keys)
    fig, axes = plt.subplots(len(examples), columns, squeeze=False,
                             figsize=(3 * columns, 2.35 * len(examples)), constrained_layout=True)
    for row, example in enumerate(examples):
        for camera, key in enumerate(camera_keys):
            rgb_ax, depth_ax, mask_ax = axes[row, 3 * camera:3 * camera + 3]
            rgb_ax.imshow(example["rgb"][key])
            depth_ax.imshow(example["depth"][camera, 0], cmap="gray", vmin=0, vmax=1)
            mask_ax.imshow(example["depth"][camera, 1], cmap="gray", vmin=0, vmax=1)
            label = key.rsplit(".", 1)[-1]
            for axis, kind in zip((rgb_ax, depth_ax, mask_ax), ("RGB", "relative depth", "valid mask"), strict=True):
                axis.set_title(f"{label}: {kind}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
            if camera == 0:
                rgb_ax.set_ylabel(f"episode {example['episode_index']} / {example['split']}\nframe {example['frame_index']}", fontsize=9)
    fig.suptitle("Depth Anything relative inverse depth: bright = near; black padding has mask 0", fontsize=13)
    fig.savefig(output / "rgb_depth_examples.png", dpi=130)
    plt.close(fig)


def write_checkpoint_csv(path, evaluations, baseline):
    columns = ["name", "step", "checkpoint_sha256", "validation_frames", "joint_mean_mae_rad",
               *[f"joint_{index}_mae_rad" for index in range(1, 7)], "gripper_mae_mm",
               "joint_max_error_rad", "gripper_max_error_mm", "validation_loss"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in [{"name": "hold_current_state_baseline", **baseline}, *evaluations]:
            row = {key: entry.get(key, "") for key in columns}
            row.update({f"joint_{index + 1}_mae_rad": value for index, value in enumerate(entry["joint_mae_rad"])})
            writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="New report directory; existing paths are refused")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--roundtrip-episodes", nargs="+", type=int)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.latency_samples < 10 or min(args.warmup, args.batch_size, args.threads) < 1:
        parser.error("Use >=10 latency samples and positive warmup/batch-size/threads")
    return args


def main(argv=None):
    args = parse_args(argv)
    data_dir, run_dir, output = [Path(value).resolve() for value in (args.data, args.run, args.output)]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report path: {output}")
    torch.set_num_threads(args.threads)
    manifest_path, run_path = data_dir / "manifest.json", run_dir / "run.json"
    manifest_sha256, run_sha256 = sha256_file(manifest_path), sha256_file(run_path)
    manifest = load_manifest(data_dir)
    run = json.loads(run_path.read_text())
    contract = validate_contract(manifest, run, manifest_sha256)
    code = audit_source_code(manifest, run)
    source = LeRobotSource(manifest["source"]["root"], manifest["camera_keys"])
    indices = manifest["source"]["selected_episode_indices"]
    fingerprint = source.fingerprint(indices)
    if fingerprint != manifest["source"]:
        raise ValueError("Original source dataset fingerprint changed after conversion")
    native = {}
    for entry in manifest["episodes"]:
        native[entry["episode_index"]] = verify_episode(data_dir / entry["path"], source,
                                                        entry["episode_index"], manifest)
    statistics = fit_statistics(data_dir, select_episodes(manifest, "train"))
    if statistics != run["statistics"]:
        raise ValueError("Training statistics do not equal a fresh train-only fit")
    validation = select_episodes(manifest, "validation")
    actions = np.concatenate([native[item["episode_index"]]["action"] for item in validation])
    states = np.concatenate([native[item["episode_index"]]["state"] for item in validation])
    baseline = action_error_summary(states, actions)
    dataset = AirbotDataset(data_dir, validation, run["statistics"], run["vocabulary"],
                           horizon=run["config"]["horizon"])
    if len(dataset) != len(actions):
        raise ValueError("Validation dataset frames differ from the native held-out episodes")
    native_all = {index: native[index]["action"] if index in native else source.load_episode(index)["action"]
                  for index in sorted(source.episodes)}
    limits = {"joint_zero_based_index": 1, "joint_name": manifest["action_names"][1],
              "reference_upper_rad": 0.17, "reference_status": "reported limit discrepancy; not an enforced limit",
              "original_source_all_episodes": episode_joint_limit_summary(native_all),
              "converted_selection": episode_joint_limit_summary({index: native[index]["action"] for index in indices}),
              "held_out_validation": episode_joint_limit_summary({item["episode_index"]: native[item["episode_index"]]["action"]
                                                                  for item in validation}), "clipping_applied": False}
    print(json.dumps({"stage": "dataset_verified", "episodes": len(indices), "frames": manifest["total_frames"],
                      "source_unchanged": True, "changed_code": code["changed_files"]}), flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", manifest["depth_config"]["cublas_workspace_config"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    paths = sorted(run_dir.glob("step_*.pt"))
    expected_steps = list(range(run["config"]["checkpoint_every"], run["completed_steps"] + 1,
                                run["config"]["checkpoint_every"]))
    actual_steps = []
    for path in paths:
        match = re.fullmatch(r"step_([0-9]{6,})\.pt", path.name)
        if match is None:
            raise ValueError(f"Unexpected retained checkpoint filename: {path.name}")
        actual_steps.append(int(match.group(1)))
    if actual_steps != expected_steps:
        raise ValueError("Retained checkpoints do not cover every configured checkpoint interval")
    if not (run_dir / "best.pt").is_file() or not (run_dir / "latest.pt").is_file():
        raise FileNotFoundError("Training must retain best.pt and latest.pt")
    evaluations = []
    for path in [*paths, run_dir / "best.pt"]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, manifest, run, manifest_sha256)
        if path.name.startswith("step_") and payload["step"] != int(path.stem.split("_")[1]):
            raise ValueError("Retained checkpoint filename and saved step differ")
        policy = AirbotDepthPolicy(path, device=args.device, local_files_only=True)
        result = {"name": path.name, "step": payload["step"], "checkpoint_sha256": policy.checkpoint_sha256,
                  "validation_loss": float(payload["validation_loss"]),
                  **evaluate_first_actions(policy, dataset, actions, args.batch_size)}
        evaluations.append(result)
        print(json.dumps({"stage": "checkpoint_evaluated", **result}), flush=True)
        del payload
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(latest, manifest, run, manifest_sha256)
    if latest["step"] != run["completed_steps"]:
        raise ValueError("latest.pt is not the final training step")
    del latest
    metrics = load_metrics(run_dir, run["completed_steps"])
    best_validation_loss = min(item["validation_loss"] for item in metrics if "validation_loss" in item)
    if evaluations[-1]["validation_loss"] != best_validation_loss or run["best_validation_loss"] != best_validation_loss:
        raise ValueError("best.pt/run do not match the best recorded validation loss")
    roundtrip, examples = audit_roundtrip(policy, source, data_dir, manifest,
                                         selected_roundtrip_indices(manifest, args.roundtrip_episodes), 1e-6)
    latency = measure_latency(policy, examples, args.latency_samples, args.warmup)
    if sha256_file(manifest_path) != manifest_sha256 or sha256_file(run_path) != run_sha256:
        raise ValueError("Manifest/run changed during audit")
    report = {"schema_version": 1, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
              "robot_executed": False, "evaluation_scope": "offline recorded-observation first-action prediction",
              "not_measured": ["real_robot_task_success_rate", "closed_loop_behavior", "recovery_from_policy_errors"],
              "data": str(data_dir), "run": str(run_dir), "manifest_sha256": manifest_sha256,
              "run_sha256": run_sha256, "config": vars(args), "environment": environment_identity(args.device),
              "dataset": {**contract, "original_source_fingerprint_verified": True,
                          "native_fields_exact_match_all_selected_episodes": True,
                          "depth_dtype_shape_range_masks_verified": True, "video_pts_alignment_verified": True,
                          "train_only_statistics_verified": True, "fps": manifest["fps"],
                          "camera_keys": manifest["camera_keys"], "prompt": manifest["prompt"],
                          "state_names": manifest["state_names"], "action_names": manifest["action_names"]},
              "source_code": code, "source_joint_limit_discrepancy": limits,
              "validation": {"measurement": "First action at each recorded held-out observation; no robot execution. "
                                              "Errors are weighted by frames, six-joint mean in radians and gripper in millimeters.",
                             "baseline": {"name": "hold_current_state", **baseline}, "checkpoints": evaluations},
              "roundtrip": roundtrip, "latency": latency,
              "training_complete_steps": run["completed_steps"],
              "formal_50000_step_training": run["completed_steps"] == 50000,
              "latest_checkpoint_sha256": sha256_file(run_dir / "latest.pt")}
    output.mkdir(parents=True, exist_ok=False)
    write_checkpoint_csv(output / "checkpoints.csv", evaluations, baseline)
    write_figures(output, metrics, examples, manifest["camera_keys"])
    sample = next((item for item in examples if item["split"] == "validation"), examples[0])
    with (output / "sample_observation.npz").open("xb") as stream:
        np.savez_compressed(stream, state=sample["state"], **sample["rgb"])
    report["sample_observation"] = {key: sample[key] for key in ("episode_index", "frame_index", "timestamp", "split")}
    report["sample_observation"].update(camera_keys=manifest["camera_keys"],
                                        path=str(output / "sample_observation.npz"),
                                        purpose="Reproducible native state plus RGB input for the offline inference CLI")
    report["artifacts"] = {name: {"path": str(output / name), "sha256": sha256_file(output / name)}
                           for name in ("checkpoints.csv", "loss_curve.png", "rgb_depth_examples.png", "sample_observation.npz")}
    atomic_json(output / "report.json", report)
    print(json.dumps({"status": "passed", "report": str(output / "report.json"),
                      "roundtrip_depth_maxdiff": roundtrip["depth_max_abs_difference"],
                      "roundtrip_action_maxdiff": roundtrip["action_chunk_max_abs_difference"],
                      "latency_mean_ms": latency["mean_ms"], "latency_p95_ms": latency["p95_ms"],
                      "robot_executed": False}), flush=True)


if __name__ == "__main__":
    main()
