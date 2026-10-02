"""Audit the unchanged 200-episode AIRBOT experiment continued from 50k to 100k.

Parent checkpoints retain their original bytes and metadata. Continuation
checkpoints are validated against their actual new run, without relabelling
the parent's random initialization or its selected best checkpoint.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airbot_depth import audit as original
from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.policy import AirbotDepthPolicy


RESTORED_FIELDS = ("model", "optimizer", "sampler_rng", "torch_rng", "python_rng", "numpy_rng", "cuda_rng")
UNCHANGED_FIELDS = (
    "manifest", "manifest_sha256", "source", "environment", "model_config", "statistics", "vocabulary",
    "depth_config", "depth_provenance", "action_semantics", "image_input_precision",
    "train_episodes", "validation_episodes", "train_frames", "validation_frames",
)


def verify_run_chain(run, parent, parent_audit, manifest, manifest_sha256, identities):
    if (run.get("status") != "complete" or run.get("initialization") != "resume_full_training_state"
            or run.get("root_initialization") != "all_policy_parameters_random"
            or run.get("start_step") != 50000 or run.get("completed_steps") != 100000
            or run.get("config", {}).get("steps") != 100000):
        raise ValueError("Expected a completed, full-state 50000-to-100000 continuation")
    # The original contract applies to the real parent, whose initialization
    # actually was random; never rewrite the resumed run to satisfy it.
    contract = original.validate_contract(manifest, parent, manifest_sha256)
    if (parent["completed_steps"] != 50000 or not contract["formal_180_train_20_validation"]
            or manifest["fps"] != 25.0):
        raise ValueError("Parent must be the original complete 200-episode 180/20 25Hz 50000-step run")
    resume = run.get("resume", {})
    if resume.get("schema_version") != 1 or resume.get("start_step") != 50000 or resume.get("target_step") != 100000:
        raise ValueError("Continuation resume boundary is invalid")
    for name, actual in identities.items():
        if resume.get(name) != actual:
            raise ValueError(f"Parent identity mismatch: {name}")
    if (parent_audit.get("status") != "passed" or parent_audit.get("formal_50000_step_training") is not True
            or parent_audit.get("training_complete_steps") != 50000
            or parent_audit.get("run_sha256") != identities["parent_run_sha256"]
            or parent_audit.get("manifest_sha256") != manifest_sha256
            or parent_audit.get("latest_checkpoint_sha256") != identities["parent_checkpoint_sha256"]):
        raise ValueError("Parent audit does not verify this exact original run/checkpoint")
    if (parent_audit.get("formal_100000_step_training", False) is not False
            or parent_audit.get("source_code", {}).get("all_saved_files_unchanged") is not True
            or parent_audit.get("roundtrip", {}).get("passed") is not True
            or parent_audit["roundtrip"].get("runtime_depth_provenance_exact_match") is not True):
        raise ValueError("Parent audit lacks the original unchanged-source and RGB roundtrip evidence")
    parent_best = [item for item in parent_audit.get("validation", {}).get("checkpoints", [])
                   if item.get("name") == "best.pt"]
    if (len(parent_best) != 1 or parent_best[0]["checkpoint_sha256"] != identities["parent_best_checkpoint_sha256"]
            or parent_best[0]["step"] != resume.get("parent_best_step")
            or parent_best[0]["validation_loss"] != resume.get("parent_best_validation_loss")):
        raise ValueError("Parent global-best identity differs from its passed audit")
    for name in UNCHANGED_FIELDS:
        if run.get(name) != parent.get(name):
            raise ValueError(f"Continuation changed a frozen training input: {name}")
    for name in set(run["config"]) | set(parent["config"]):
        if name not in {"steps", "output"} and run["config"].get(name) != parent["config"].get(name):
            raise ValueError(f"Continuation changed the parent hyperparameter: {name}")
    required_fields = set(RESTORED_FIELDS) | {"best_validation_loss", "numerical_flags"}
    if not required_fields <= set(resume.get("restored_fields", [])):
        raise ValueError("Resume metadata omits required full-training-state fields")
    return contract


def verify_optimizer_step(payload, expected_step):
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("Checkpoint is missing nonempty AdamW optimizer state")
    parameters = [key for group in optimizer["param_groups"] for key in group["params"]]
    if (len(parameters) != len(set(parameters)) or set(parameters) != set(optimizer["state"])
            or len(parameters) != len(payload["model"])):
        raise ValueError("Optimizer state does not cover every policy parameter exactly once")
    steps = []
    for key, parameter in zip(parameters, payload["model"].values(), strict=True):
        state = optimizer["state"][key]
        step = state.get("step")
        if not isinstance(step, (int, float, torch.Tensor)) or torch.as_tensor(step).numel() != 1:
            raise ValueError("Optimizer parameter has no scalar step")
        value = float(torch.as_tensor(step).item())
        if not np.isfinite(value) or value != expected_step:
            raise ValueError(f"Optimizer step differs from checkpoint step {expected_step}")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid AdamW momentum tensor: {name}")
        steps.append(value)
    return {"count": len(steps), "min": min(steps), "max": max(steps)}


def verify_restoration(run, parent_checkpoint):
    from scripts.resume_airbot_depth import NUMERICAL_FLAGS, state_digest, training_state_digests

    if parent_checkpoint.get("step") != 50000:
        raise ValueError("Resume checkpoint must be exactly optimizer step 50000")
    summary = verify_optimizer_step(parent_checkpoint, 50000)
    if any(name not in parent_checkpoint for name in RESTORED_FIELDS):
        raise ValueError("Parent checkpoint is missing required RNG/training state")
    if parent_checkpoint["environment"].get("device") == "cuda" and not parent_checkpoint["cuda_rng"]:
        raise ValueError("CUDA parent checkpoint is missing CUDA RNG states")
    digests = training_state_digests(parent_checkpoint)
    proof = run.get("resume_verification", {})
    if (proof.get("schema_version") != 1 or proof.get("all_restored_exact") is not True
            or proof.get("saved_state_digests") != digests or proof.get("restored_state_digests") != digests):
        raise ValueError("Recorded restored state does not exactly match the actual parent checkpoint")
    for key in ("optimizer_checkpoint", "optimizer_after_restore"):
        if proof.get(key) != summary:
            raise ValueError(f"Resume optimizer restoration proof differs: {key}")
    if proof.get("optimizer_after_first_step") != {"count": summary["count"], "min": 50001.0, "max": 50001.0}:
        raise ValueError("First continuation optimizer update did not advance from 50000 to 50001")
    flags = proof.get("runtime_flags", {})
    expected_flags = {name: parent_checkpoint["environment"][name] for name in NUMERICAL_FLAGS}
    if (flags.get("exact_match") is not True or flags.get("saved") != flags.get("restored")
            or flags.get("saved") != expected_flags):
        raise ValueError("Numerical runtime flags were not restored exactly")
    for name, value in flags["saved"].items():
        if parent_checkpoint["environment"].get(name) != value:
            raise ValueError(f"Restored numerical flag differs from parent: {name}")
    generator = torch.Generator()
    generator.set_state(parent_checkpoint["sampler_rng"])
    before = state_digest(generator.get_state())
    expected = torch.randint(run["train_frames"], (run["config"]["batch_size"],), generator=generator).tolist()
    after = state_digest(generator.get_state())
    first = proof.get("first_batch", {})
    if (first.get("global_step") != 50001 or first.get("exact_match") is not True
            or first.get("expected_indices") != expected or first.get("actual_indices") != expected
            or first.get("sampler_rng_before_sha256") != before or first.get("sampler_rng_after_sha256") != after
            or first.get("expected_sampler_rng_after_sha256") != after):
        raise ValueError("First resumed minibatch differs from the restored parent's sampler sequence")
    return {"full_state_digests": digests, "all_restored_exact": True,
            "optimizer_at_resume": summary, "optimizer_after_first_step": proof["optimizer_after_first_step"],
            "first_batch_recomputed": True, "runtime_flags_restored": True}


def verify_metrics(run_dir, parent_dir, run):
    parent_bytes = (parent_dir / "metrics.jsonl").read_bytes()
    current_bytes = (run_dir / "metrics.jsonl").read_bytes()
    resume = run["resume"]
    if (not parent_bytes.endswith(b"\n") or not current_bytes.startswith(parent_bytes)
            or len(parent_bytes) != resume["parent_metrics_bytes"]
            or hashlib.sha256(parent_bytes).hexdigest() != resume["parent_metrics_sha256"]):
        raise ValueError("Continuation metrics do not preserve the exact parent byte prefix")
    parent_records = original.load_metrics(parent_dir, run["start_step"])
    if len(parent_records) != resume["parent_metrics_records"]:
        raise ValueError("Parent metrics record count differs from resume provenance")
    metrics = original.load_metrics(run_dir, run["completed_steps"])
    suffix = metrics[len(parent_records):]
    start, end = run["start_step"], run["completed_steps"]
    expected = [step for step in range(start + 1, end + 1)
                if step == start + 1 or step == end or step % 25 == 0
                or step % run["config"]["eval_every"] == 0 or step % run["config"]["checkpoint_every"] == 0]
    if [item["step"] for item in suffix] != expected:
        raise ValueError("Continuation metrics omit or duplicate expected logged optimizer steps")
    for item in suffix:
        if ((item["step"] % run["config"]["eval_every"] == 0
             or item["step"] % run["config"]["checkpoint_every"] == 0 or item["step"] == end)
                and "validation_loss" not in item):
            raise ValueError("Continuation metrics omit a required held-out validation")
    return metrics, {"parent_prefix_bytes": len(parent_bytes), "parent_prefix_sha256": resume["parent_metrics_sha256"],
                     "parent_records": len(parent_records), "continuation_records": len(suffix),
                     "exact_parent_prefix": True, "logged_step_coverage_verified": True,
                     "metrics_sha256": sha256_file(run_dir / "metrics.jsonl")}


def checkpoint_origin(payload, digest, run, parent, manifest, manifest_sha256, parent_identities):
    step = payload.get("step")
    if type(step) is not int or step < 1 or step > 100000:
        raise ValueError("Checkpoint has an invalid global optimizer step")
    if step <= 50000:
        if parent_identities.get(digest) != step:
            raise ValueError("Parent-phase checkpoint is not an exact inherited audited checkpoint")
        origin, source = "parent", parent
    else:
        origin, source = "continuation", run
        for key in ("initialization", "root_initialization", "start_step", "resume", "driver_source", "resume_verification"):
            if payload.get(key) != run.get(key):
                raise ValueError(f"New checkpoint differs from continuation metadata: {key}")
    original.validate_checkpoint(payload, manifest, source, manifest_sha256)
    verify_optimizer_step(payload, step)
    return {"phase": origin, "origin_run_target_steps": source["config"]["steps"],
            "origin_run_sha256": run["resume"]["parent_run_sha256"] if origin == "parent" else sha256_file(Path(run["config"]["output"]) / "run.json")}


def load_chain(run_dir, manifest, manifest_sha256):
    run = json.loads((run_dir / "run.json").read_text())
    resume = run["resume"]
    parent_dir = Path(resume["parent_run"]).resolve(strict=True)
    parent_path, audit_path = parent_dir / "run.json", Path(resume["parent_audit"]).resolve(strict=True)
    latest_path, best_path = Path(resume["parent_checkpoint"]).resolve(strict=True), Path(resume["parent_best_checkpoint"]).resolve(strict=True)
    if (latest_path != parent_dir / "latest.pt" or best_path != parent_dir / "best.pt"
            or Path(run["config"]["output"]).resolve() != run_dir):
        raise ValueError("Resume paths do not identify the declared parent/new run")
    identities = {"parent_run_sha256": sha256_file(parent_path), "parent_audit_sha256": sha256_file(audit_path),
                  "parent_checkpoint_sha256": sha256_file(latest_path), "parent_best_checkpoint_sha256": sha256_file(best_path)}
    parent, parent_audit = json.loads(parent_path.read_text()), json.loads(audit_path.read_text())
    contract = verify_run_chain(run, parent, parent_audit, manifest, manifest_sha256, identities)
    driver = run.get("driver_source", {})
    if set(driver) != {"scripts/resume_airbot_depth.py"} or sha256_file(ROOT / "scripts/resume_airbot_depth.py") != driver["scripts/resume_airbot_depth.py"]:
        raise ValueError("Continuation driver differs from its recorded source")
    original.audit_source_code(manifest, parent)
    code = original.audit_source_code(manifest, run)
    code["continuation_audit_source_sha256"] = sha256_file(__file__)
    parent_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    original.validate_checkpoint(parent_checkpoint, manifest, parent, manifest_sha256)
    restored = verify_restoration(run, parent_checkpoint)
    del parent_checkpoint
    inherited = run.get("inherited_checkpoints", [])
    expected_paths = {f"step_{step:06d}.pt" for step in range(10000, 50001, 10000)} | {"parent_best.pt"}
    if len(inherited) != len(expected_paths) or {item["path"] for item in inherited} != expected_paths:
        raise ValueError("Inherited checkpoints do not cover the five parent snapshots and original best")
    audited = {item["checkpoint_sha256"]: item["step"] for item in parent_audit["validation"]["checkpoints"]}
    parent_identities = {}
    for item in inherited:
        expected_source = parent_dir / ("best.pt" if item["path"] == "parent_best.pt" else item["path"])
        if (Path(item["source_path"]).resolve() != expected_source
                or sha256_file(expected_source) != item["sha256"]
                or sha256_file(run_dir / item["path"]) != item["sha256"]
                or audited.get(item["sha256"]) != item["step"]):
            raise ValueError(f"Inherited checkpoint bytes differ from parent audit: {item['path']}")
        parent_identities[item["sha256"]] = item["step"]
    metrics, metrics_proof = verify_metrics(run_dir, parent_dir, run)
    continuation = {"schema_version": 1, "verified": True, "start_step": 50000, "end_step": 100000,
                    "initialization": run["initialization"], "root_initialization": run["root_initialization"],
                    "parent_run": str(parent_dir), "parent_audit": str(audit_path), **identities,
                    "parent_run_target_steps": 50000, "driver_source": driver,
                    "restoration": restored, "metrics": metrics_proof,
                    "inherited_checkpoints": inherited, "checkpoint_origins": []}
    return run, parent, contract, code, metrics, parent_identities, continuation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", help="Optional check against the frozen run's dataset path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--roundtrip-episodes", type=int, nargs="+")
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.warmup) < 1 or args.latency_samples < 10:
        parser.error("Use positive threads/batch/warmup and at least 10 latency samples")
    run_dir, output = Path(args.run).resolve(strict=True), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
    preliminary = json.loads((run_dir / "run.json").read_text())
    data_dir = Path(preliminary["config"]["data"]).resolve(strict=True)
    if args.data and Path(args.data).resolve() != data_dir:
        raise ValueError("--data differs from the unchanged run dataset")
    torch.set_num_threads(args.threads)
    manifest_sha256 = sha256_file(data_dir / "manifest.json")
    run_sha256 = sha256_file(run_dir / "run.json")
    manifest = original.load_manifest(data_dir)
    run, parent, contract, code, metrics, parent_ids, continuation = load_chain(run_dir, manifest, manifest_sha256)
    source = original.LeRobotSource(manifest["source"]["root"], manifest["camera_keys"])
    if source.fingerprint(manifest["source"]["selected_episode_indices"]) != manifest["source"]:
        raise ValueError("Original 200-episode source fingerprint changed")
    native = {item["episode_index"]: original.verify_episode(data_dir / item["path"], source, item["episode_index"], manifest)
              for item in manifest["episodes"]}
    if original.fit_statistics(data_dir, original.select_episodes(manifest, "train")) != run["statistics"]:
        raise ValueError("Frozen statistics differ from a fresh train-only fit")
    validation = original.select_episodes(manifest, "validation")
    actions = np.concatenate([native[item["episode_index"]]["action"] for item in validation])
    states = np.concatenate([native[item["episode_index"]]["state"] for item in validation])
    baseline = original.action_error_summary(states, actions)
    dataset = original.AirbotDataset(data_dir, validation, run["statistics"], run["vocabulary"], horizon=run["config"]["horizon"])
    if len(dataset) != len(actions):
        raise ValueError("Validation frame count differs from native episodes")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", manifest["depth_config"]["cublas_workspace_config"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    names = [f"step_{step:06d}.pt" for step in range(10000, 100001, 10000)]
    if sorted(path.name for path in run_dir.glob("step_*.pt")) != names:
        raise ValueError("Continuation must retain all ten 10000-step checkpoints")
    metric_by_step = {item["step"]: item for item in metrics}
    global_best = min((item for item in metrics if "validation_loss" in item), key=lambda item: (item["validation_loss"], item["step"]))
    segment_best = min((item for item in metrics if item["step"] > 50000 and "validation_loss" in item), key=lambda item: (item["validation_loss"], item["step"]))
    if (run.get("best_validation_loss") != global_best["validation_loss"] or run.get("best_step") != global_best["step"]
            or run.get("best_continuation_validation_loss") != segment_best["validation_loss"]
            or run.get("best_continuation_step") != segment_best["step"]):
        raise ValueError("Run's global/continuation best records differ from the complete metrics history")
    evaluations = []
    final_state_digests = None
    for name in [*names, "best_continuation.pt", "best.pt", "latest.pt"]:
        path = run_dir / name
        digest = sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        origin = checkpoint_origin(payload, digest, run, parent, manifest, manifest_sha256, parent_ids)
        step = payload["step"]
        expected_step = (int(name[5:-3]) if name.startswith("step_") else
                         global_best["step"] if name == "best.pt" else segment_best["step"] if name == "best_continuation.pt" else 100000)
        if step != expected_step or payload["validation_loss"] != metric_by_step[step]["validation_loss"]:
            raise ValueError(f"Checkpoint identity/metric differs from its declared role: {name}")
        record = {"name": name, "checkpoint_sha256": digest, "step": step, **origin}
        continuation["checkpoint_origins"].append(record)
        if name in {"step_100000.pt", "latest.pt"}:
            from scripts.resume_airbot_depth import training_state_digests
            digests = training_state_digests(payload)
            if name == "step_100000.pt":
                final_state_digests = digests
            elif digests != final_state_digests:
                raise ValueError("Latest checkpoint differs from the retained 100000-step full training state")
        if name == "latest.pt":
            continuation["final_optimizer"] = verify_optimizer_step(payload, 100000)
            continuation["final_full_state_digests"] = final_state_digests
            del payload
            continue
        policy = AirbotDepthPolicy(path, device=args.device, local_files_only=True)
        if policy.checkpoint_sha256 != digest:
            raise ValueError("Checkpoint changed between metadata validation and inference loading")
        evaluated = {**record, "validation_loss": float(payload["validation_loss"]),
                     **original.evaluate_first_actions(policy, dataset, actions, args.batch_size)}
        evaluations.append(evaluated)
        print(json.dumps({"stage": "checkpoint_evaluated", **evaluated}), flush=True)
        del payload
    # The last evaluated policy is the actual global best, possibly inherited.
    indices = original.selected_roundtrip_indices(manifest, args.roundtrip_episodes)
    if not set(indices) & {item["episode_index"] for item in validation}:
        indices.append(validation[0]["episode_index"])
    roundtrip, examples = original.audit_roundtrip(policy, source, data_dir, manifest, indices, 1e-6)
    latency = original.measure_latency(policy, examples, args.latency_samples, args.warmup)
    limits = {"joint_zero_based_index": 1, "joint_name": manifest["action_names"][1], "reference_upper_rad": 0.17,
              "reference_status": "reported limit discrepancy; not an enforced limit", "clipping_applied": False,
              "original_source_all_episodes": original.episode_joint_limit_summary({index: values["action"] for index, values in native.items()}),
              "held_out_validation": original.episode_joint_limit_summary({item["episode_index"]: native[item["episode_index"]]["action"] for item in validation})}
    if sha256_file(data_dir / "manifest.json") != manifest_sha256 or sha256_file(run_dir / "run.json") != run_sha256:
        raise ValueError("Manifest or run changed during continuation audit")
    for item in continuation["checkpoint_origins"]:
        if sha256_file(run_dir / item["name"]) != item["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during continuation audit")
    report = {"schema_version": 1, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
              "data": str(data_dir), "run": str(run_dir), "manifest_sha256": manifest_sha256, "run_sha256": run_sha256,
              "config": vars(args), "environment": original.environment_identity(args.device),
              "initialization": run["initialization"], "root_initialization": run["root_initialization"],
              "continuation": continuation, "source_code": code, "source_joint_limit_discrepancy": limits,
              "dataset": {**contract, "original_source_fingerprint_verified": True, "native_fields_exact_match_all_selected_episodes": True,
                          "depth_dtype_shape_range_masks_verified": True, "video_pts_alignment_verified": True, "train_only_statistics_verified": True,
                          **{key: manifest[key] for key in ("fps", "camera_keys", "prompt", "state_names", "action_names")}},
              "validation": {"measurement": "Recorded held-out first-action errors; not robot task success.",
                             "baseline": {"name": "hold_current_state", **baseline}, "checkpoints": evaluations},
              "training_complete_steps": 100000, "formal_50000_step_training": False, "formal_100000_step_training": True,
              "latest_checkpoint_sha256": sha256_file(run_dir / "latest.pt"), "roundtrip": roundtrip, "latency": latency,
              "evaluation_scope": "offline recorded-observation first-action prediction", "robot_executed": False,
              "not_measured": ["real_robot_task_success_rate", "closed_loop_behavior", "recovery_from_policy_errors"]}
    output.mkdir(parents=True, exist_ok=False)
    original.write_checkpoint_csv(output / "checkpoints.csv", evaluations, baseline)
    original.write_figures(output, metrics, examples, manifest["camera_keys"])
    sample = next(item for item in examples if item["split"] == "validation")
    with (output / "sample_observation.npz").open("xb") as stream:
        np.savez_compressed(stream, state=sample["state"], **sample["rgb"])
    report["sample_observation"] = {key: sample[key] for key in ("episode_index", "frame_index", "timestamp", "split")}
    report["sample_observation"].update(camera_keys=manifest["camera_keys"], path=str(output / "sample_observation.npz"))
    report["artifacts"] = {name: {"path": str(output / name), "sha256": sha256_file(output / name)}
                           for name in ("checkpoints.csv", "loss_curve.png", "rgb_depth_examples.png", "sample_observation.npz")}
    atomic_json(output / "report.json", report)
    print(json.dumps({"status": "passed", "report": str(output / "report.json"), "checkpoints": len(evaluations),
                      "global_best_actual_step": global_best["step"], "continuation_verified": True, "robot_executed": False}), flush=True)


if __name__ == "__main__":
    main()
