"""Continue an audited AIRBOT run with its full optimizer and RNG state.

The frozen airbot_depth package is reused without modification. --steps is the
global target step, not an additional step count. Existing outputs are refused.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airbot_depth.audit import load_metrics
from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.data import AirbotDataset, fit_statistics, load_manifest
from airbot_depth.model import AirbotPolicyModel, batch_loss
from airbot_depth.train import (
    environment_identity, evaluate_loss, record_training_failure, save_checkpoint,
    source_identity, to_device, validate_training_manifest,
)
from depth_policy.data import build_vocabulary


STATE_FIELDS = ("model", "optimizer", "sampler_rng", "torch_rng", "python_rng", "numpy_rng", "cuda_rng")
NUMERICAL_FLAGS = ("deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
                   "cudnn_allow_tf32", "matmul_allow_tf32")
PARENT_FIELDS = ("schema_version", "format", "config", "model_config", "statistics", "vocabulary",
                 "depth_config", "depth_provenance", "action_semantics", "image_input_precision",
                 "manifest", "manifest_sha256", "source", "environment")


def state_digest(value):
    """Stable exact-value hash, including shape/dtype and excluding tensor device."""
    digest = hashlib.sha256()
    def update(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(value.dtype).encode() + repr(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            value = np.ascontiguousarray(item)
            digest.update(b"ndarray\0" + value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                update(key)
                update(item[key])
            digest.update(b"enddict\0")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for value in item:
                update(value)
            digest.update(b"endsequence\0")
        elif isinstance(item, np.generic):
            update(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(json.dumps(item, allow_nan=False, ensure_ascii=False).encode() + b"\0")
        else:
            raise TypeError(f"Unsupported training-state value: {type(item).__name__}")
    update(value)
    return digest.hexdigest()


def training_state_digests(payload):
    missing = set(STATE_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"Checkpoint lacks full training state: {sorted(missing)}")
    return {name: state_digest(payload[name]) for name in STATE_FIELDS}


def optimizer_step_summary(optimizer_or_state_dict):
    state = (optimizer_or_state_dict.state_dict() if isinstance(optimizer_or_state_dict, torch.optim.Optimizer)
             else optimizer_or_state_dict)
    steps = []
    for value in state["state"].values():
        step = value.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step):
            raise ValueError("Optimizer state has a missing or nonfinite step counter")
        steps.append(float(step.item()))
    return {"count": len(steps), "min": min(steps) if steps else None, "max": max(steps) if steps else None}


def capture_training_state(model, optimizer, sampler, device):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else None}


def restore_numerical_environment(saved, device, threads, expected_threads):
    if threads != expected_threads or str(device) != saved["device"]:
        raise ValueError("Resume device and thread count must match the audited parent run")
    if any(type(saved.get(name)) is not bool for name in NUMERICAL_FLAGS):
        raise ValueError("Parent run lacks explicit numerical execution flags")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(saved["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = saved["matmul_allow_tf32"]
    actual = environment_identity(device)
    if actual != saved:
        differing = [name for name in sorted(set(actual) | set(saved)) if actual.get(name) != saved.get(name)]
        raise ValueError(f"Resume runtime differs from the audited parent: {differing}")
    return {"saved": {name: saved[name] for name in NUMERICAL_FLAGS},
            "restored": {name: actual[name] for name in NUMERICAL_FLAGS}, "exact_match": True}


def restore_training_state(payload, model, optimizer, sampler, device):
    """Restore only after construction/data loading, so initialization consumes no saved RNG."""
    expected = training_state_digests(payload)
    before = optimizer_step_summary(optimizer)
    saved_optimizer = payload["optimizer"]
    template = optimizer.state_dict()
    if saved_optimizer.get("param_groups") != template["param_groups"]:
        raise ValueError("Saved AdamW parameter groups/hyperparameters differ from the inherited optimizer")
    parameter_ids = [identifier for group in template["param_groups"] for identifier in group["params"]]
    if set(saved_optimizer.get("state", {})) != set(parameter_ids):
        raise ValueError("Checkpoint does not contain AdamW state for every policy parameter")
    summary = optimizer_step_summary(saved_optimizer)
    if summary["min"] != payload["step"] or summary["max"] != payload["step"]:
        raise ValueError("AdamW step counters do not equal the parent checkpoint step")
    parameters = list(model.parameters())
    for identifier, parameter in zip(parameter_ids, parameters, strict=True):
        state = saved_optimizer["state"][identifier]
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid AdamW moment: {key}")
    if torch.device(device).type == "cuda":
        if not isinstance(payload["cuda_rng"], list) or len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint lacks the complete CUDA RNG state for this runtime")
    elif payload["cuda_rng"] is not None:
        raise ValueError("CPU continuation cannot discard saved CUDA RNG state")
    for name in ("sampler_rng", "torch_rng"):
        if not isinstance(payload[name], torch.Tensor) or payload[name].dtype != torch.uint8 or payload[name].ndim != 1:
            raise ValueError(f"Invalid {name}")
    model.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint model contains nonfinite parameters")
    optimizer.load_state_dict(saved_optimizer)
    sampler.set_state(payload["sampler_rng"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    actual = training_state_digests(capture_training_state(model, optimizer, sampler, device))
    if actual != expected:
        raise ValueError("Training state differs after full restoration")
    return {"schema_version": 1, "saved_state_digests": expected, "restored_state_digests": actual,
            "all_restored_exact": True, "optimizer_before_restore": before,
            "optimizer_checkpoint": summary, "optimizer_after_restore": optimizer_step_summary(optimizer)}


def validate_parent_metadata(parent, audit, latest, best, *, run_sha, latest_sha, best_sha,
                             manifest, manifest_sha, current_source, parent_run):
    if parent.get("status") != "complete" or parent.get("initialization") != "all_policy_parameters_random":
        raise ValueError("This driver requires a completed original randomly initialized parent run")
    if audit.get("schema_version") != 1 or audit.get("status") != "passed":
        raise ValueError("Parent audit must have passed")
    if (audit.get("run_sha256") != run_sha or Path(audit["run"]).resolve() != Path(parent_run).resolve()
            or audit.get("latest_checkpoint_sha256") != latest_sha):
        raise ValueError("Parent audit does not bind the current parent run/latest checkpoint")
    if (parent.get("manifest_sha256") != manifest_sha or audit.get("manifest_sha256") != manifest_sha
            or parent.get("manifest") != manifest):
        raise ValueError("Parent run/audit dataset manifest changed")
    if parent.get("source") != current_source:
        raise ValueError("Frozen policy/data/training source changed after the parent run")
    if audit.get("source_code", {}).get("all_saved_files_unchanged") is not True:
        raise ValueError("Parent audit did not verify unchanged training sources")
    if (audit.get("roundtrip", {}).get("passed") is not True
            or audit["roundtrip"].get("runtime_depth_provenance_exact_match") is not True):
        raise ValueError("Parent audit lacks the verified RGB/depth roundtrip")
    for payload in (latest, best):
        for key in PARENT_FIELDS:
            if payload.get(key) != parent.get(key):
                raise ValueError(f"Parent checkpoint differs from run metadata: {key}")
    step = parent.get("completed_steps")
    if (type(step) is not int or step < 1 or latest.get("step") != step
            or parent["config"]["steps"] != step or audit.get("training_complete_steps") != step):
        raise ValueError("Latest checkpoint is not the audited completed parent step")
    if (not np.isfinite(parent.get("best_validation_loss", float("nan")))
            or latest.get("best_validation_loss") != parent["best_validation_loss"]
            or best.get("validation_loss") != parent["best_validation_loss"]
            or best.get("best_validation_loss") != parent["best_validation_loss"]):
        raise ValueError("Parent global best validation history is inconsistent")
    identities = [item for item in audit["validation"]["checkpoints"] if item.get("name") == "best.pt"]
    if (len(identities) != 1 or identities[0].get("checkpoint_sha256") != best_sha
            or identities[0].get("step") != best["step"]):
        raise ValueError("Parent best checkpoint is not bound by its passed audit")
    training_state_digests(latest)


def copy_exact(source, destination, expected_sha):
    if destination.exists():
        raise FileExistsError(destination)
    if sha256_file(source) != expected_sha:
        raise ValueError(f"Inherited file changed before copy: {source}")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256_file(destination) != expected_sha:
        raise ValueError(f"Inherited copy checksum mismatch: {destination}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--parent-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100000, help="Global optimizer step to reach")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    parent_dir, audit_path, output = [Path(value).resolve() for value in (args.parent_run, args.parent_audit, args.output)]
    if output.exists():
        raise FileExistsError("Continuation output must be a new directory; overwrite/resume-in-place is refused")
    if output.is_relative_to(parent_dir):
        raise ValueError("Continuation output must be separate from the parent run")
    parent_path, latest_path, best_path = parent_dir / "run.json", parent_dir / "latest.pt", parent_dir / "best.pt"
    parent = json.loads(parent_path.read_text())
    audit = json.loads(audit_path.read_text())
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    parent_sha, audit_sha, latest_sha, best_sha = [sha256_file(path) for path in (parent_path, audit_path, latest_path, best_path)]
    data_dir = Path(parent["config"]["data"]).resolve()
    if output.is_relative_to(data_dir) or data_dir.is_relative_to(output):
        raise ValueError("Continuation output must be separate from the converted dataset")
    manifest_sha = sha256_file(data_dir / "manifest.json")
    manifest = load_manifest(data_dir)
    frozen_source = source_identity()
    validate_parent_metadata(parent, audit, latest, best_payload, run_sha=parent_sha, latest_sha=latest_sha,
                             best_sha=best_sha, manifest=manifest, manifest_sha=manifest_sha,
                             current_source=frozen_source, parent_run=parent_dir)
    start_step = latest["step"]
    if args.steps <= start_step:
        raise ValueError("--steps must exceed the parent's completed global step")
    flags = restore_numerical_environment(parent["environment"], args.device, args.threads, parent["config"]["threads"])
    training, validation = validate_training_manifest(data_dir, manifest)
    if (fit_statistics(data_dir, training) != parent["statistics"]
            or build_vocabulary([{"instruction": manifest["prompt"]}]) != parent["vocabulary"]):
        raise ValueError("Training-only normalization or vocabulary differs from the parent")
    config = deepcopy(parent["config"])
    config.update(output=str(output), steps=args.steps, device=args.device, threads=args.threads)
    metrics_path = parent_dir / "metrics.jsonl"
    metrics_bytes = metrics_path.read_bytes()
    if not metrics_bytes.endswith(b"\n"):
        raise ValueError("Parent metrics prefix must end with its original newline")
    metrics = load_metrics(parent_dir, start_step)
    parent_elapsed = parent["elapsed_s"]
    if not np.isfinite(parent_elapsed) or parent_elapsed < metrics[-1]["elapsed_s"]:
        raise ValueError("Parent elapsed time is inconsistent with its metrics")
    if min(item["validation_loss"] for item in metrics if "validation_loss" in item) != parent["best_validation_loss"]:
        raise ValueError("Parent metrics do not preserve global best validation history")
    inherited = []
    audited = {item["name"]: item for item in audit["validation"]["checkpoints"]}
    interval = config["checkpoint_every"]
    for step in range(interval, start_step + 1, interval):
        name = f"step_{step:06d}.pt"
        record = audited.get(name)
        path = parent_dir / name
        if record is None or record["step"] != step or sha256_file(path) != record["checkpoint_sha256"]:
            raise ValueError(f"Parent retained checkpoint is not bound by the passed audit: {name}")
        inherited.append({"path": name, "source_path": str(path), "step": step,
                          "sha256": record["checkpoint_sha256"], "role": "retained"})
    inherited.append({"path": "parent_best.pt", "source_path": str(best_path), "step": best_payload["step"],
                      "sha256": best_sha, "role": "parent_best"})
    resume = {"schema_version": 1, "parent_run": str(parent_dir), "parent_run_sha256": parent_sha,
              "parent_audit": str(audit_path), "parent_audit_sha256": audit_sha,
              "parent_checkpoint": str(latest_path), "parent_checkpoint_sha256": latest_sha,
              "parent_best_checkpoint": str(best_path), "parent_best_checkpoint_sha256": best_sha,
              "parent_best_step": best_payload["step"], "parent_best_validation_loss": parent["best_validation_loss"],
              "parent_elapsed_s": parent_elapsed, "parent_metrics_sha256": sha256_file(metrics_path),
              "parent_metrics_bytes": len(metrics_bytes), "parent_metrics_records": len(metrics),
              "start_step": start_step, "target_step": args.steps,
              "restored_fields": [*STATE_FIELDS, "best_validation_loss", "numerical_flags"]}
    driver = {str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(__file__)}
    model = AirbotPolicyModel(**parent["model_config"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
    datasets = [AirbotDataset(data_dir, episodes, parent["statistics"], parent["vocabulary"], horizon=config["horizon"])
                for episodes in (training, validation)]
    if len(datasets[0]) != parent["train_frames"] or len(datasets[1]) != parent["validation_frames"]:
        raise ValueError("Continuation dataset frames differ from the audited parent")
    sampler = torch.Generator()
    verification = restore_training_state(latest, model, optimizer, sampler, args.device)
    verification["runtime_flags"] = flags
    expected_sampler = torch.Generator()
    expected_sampler.set_state(sampler.get_state())
    expected_first_indices = torch.randint(len(datasets[0]), (config["batch_size"],), generator=expected_sampler).tolist()
    verification["first_batch"] = {"global_step": start_step + 1, "expected_indices": expected_first_indices,
                                   "actual_indices": None, "exact_match": False,
                                   "sampler_rng_before_sha256": state_digest(sampler.get_state()),
                                   "expected_sampler_rng_after_sha256": state_digest(expected_sampler.get_state())}
    run = deepcopy(parent)
    run.update(status="running", created_at=datetime.now(timezone.utc).isoformat(), config=config,
               initialization="resume_full_training_state", root_initialization="all_policy_parameters_random",
               start_step=start_step, completed_steps=start_step, resume=resume, resume_verification=verification,
               inherited_checkpoints=inherited, driver_source=driver, best_step=best_payload["step"],
               best_continuation_step=None, best_continuation_validation_loss=None,
               segment_elapsed_s=0.0, elapsed_s=parent_elapsed)
    output.mkdir(parents=True, exist_ok=False)
    for record in inherited:
        copy_exact(Path(record["source_path"]), output / record["path"], record["sha256"])
    copy_exact(best_path, output / "best.pt", best_sha)
    copy_exact(metrics_path, output / "metrics.jsonl", resume["parent_metrics_sha256"])
    atomic_json(output / "run.json", run)
    print(json.dumps({"stage": "restored", "start_step": start_step, "target_step": args.steps,
                      "optimizer": verification["optimizer_after_restore"], "all_restored_exact": True}), flush=True)
    best, best_step = parent["best_validation_loss"], best_payload["step"]
    continuation_best, continuation_step = float("inf"), None
    del latest, best_payload
    segment_start = time.perf_counter()
    model.train()
    with record_training_failure(output, run), (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start_step + 1, args.steps + 1):
            indices = torch.randint(len(datasets[0]), (config["batch_size"],), generator=sampler).tolist()
            if step == start_step + 1:
                first = verification["first_batch"]
                first.update(actual_indices=indices, exact_match=indices == expected_first_indices,
                             sampler_rng_after_sha256=state_digest(sampler.get_state()))
                if not first["exact_match"] or first["sampler_rng_after_sha256"] != first["expected_sampler_rng_after_sha256"]:
                    raise ValueError("First continuation batch differs from the restored sampler state")
            batch = to_device(default_collate([datasets[0][index] for index in indices]), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == start_step + 1:
                verification["optimizer_after_first_step"] = optimizer_step_summary(optimizer)
                if (verification["optimizer_after_first_step"]["min"] != step
                        or verification["optimizer_after_first_step"]["max"] != step):
                    raise ValueError("AdamW did not advance each restored parameter by one step")
                atomic_json(output / "run.json", run)
            segment_elapsed = time.perf_counter() - segment_start
            record = {"step": step, "train_loss": float(loss.detach().item()),
                      "gradient_norm": float(gradient_norm.item()), "elapsed_s": parent_elapsed + segment_elapsed,
                      "segment_elapsed_s": segment_elapsed}
            retain = step % config["checkpoint_every"] == 0
            evaluate = step % config["eval_every"] == 0 or retain or step == args.steps
            if evaluate:
                validation_loss = evaluate_loss(model, datasets[1], args.device, config["batch_size"])
                record["validation_loss"] = validation_loss
                improved, segment_improved = validation_loss < best, validation_loss < continuation_best
                if improved:
                    best, best_step = validation_loss, step
                if segment_improved:
                    continuation_best, continuation_step = validation_loss, step
                payload = {key: deepcopy(run[key]) for key in PARENT_FIELDS if key != "config"}
                payload.update(config=config, **capture_training_state(model, optimizer, sampler, args.device),
                               step=step, validation_loss=validation_loss, best_validation_loss=best,
                               initialization="resume_full_training_state", root_initialization="all_policy_parameters_random",
                               start_step=start_step, resume=deepcopy(resume), resume_verification=deepcopy(verification),
                               inherited_checkpoints=inherited, driver_source=driver, best_step=best_step,
                               best_continuation_step=continuation_step, best_continuation_validation_loss=continuation_best)
                if retain:
                    destination = output / f"step_{step:06d}.pt"
                    if destination.exists():
                        raise FileExistsError(f"Refusing to overwrite retained checkpoint: {destination}")
                    save_checkpoint(destination, payload)
                save_checkpoint(output / "latest.pt", payload)
                if improved:
                    save_checkpoint(output / "best.pt", payload)
                if segment_improved:
                    save_checkpoint(output / "best_continuation.pt", payload)
            if evaluate or step % 25 == 0 or step == start_step + 1:
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
        if source_identity() != frozen_source or sha256_file(__file__) != next(iter(driver.values())):
            raise ValueError("Frozen package or continuation driver changed during training")
        if sha256_file(data_dir / "manifest.json") != manifest_sha:
            raise ValueError("Data manifest changed during continuation")
    elapsed = time.perf_counter() - segment_start
    run.update(status="complete", completed_steps=args.steps, completed_at=datetime.now(timezone.utc).isoformat(),
               best_validation_loss=best, best_step=best_step,
               best_continuation_step=continuation_step, best_continuation_validation_loss=continuation_best,
               segment_elapsed_s=elapsed, elapsed_s=parent_elapsed + elapsed)
    atomic_json(output / "run.json", run)


if __name__ == "__main__":
    main()
