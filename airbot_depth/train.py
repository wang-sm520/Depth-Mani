"""Train a randomly initialized policy on a completed Airbot depth conversion."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.data import AirbotDataset, fit_statistics, load_manifest, select_episodes
from airbot_depth.depth import canonical_depth_config
from airbot_depth.model import AirbotPolicyModel, CHECKPOINT_FORMAT, IMAGE_INPUT_PRECISION, batch_loss
from depth_policy.data import build_vocabulary, encode_language


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "airbot_depth").glob("*.py"))
    files += [root / "depth_policy" / name for name in ("model.py", "data.py")]
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def environment_identity(device):
    selected = torch.device(device)
    gpu_name = None
    if selected.type == "cuda":
        gpu_name = torch.cuda.get_device_name(selected)
    return {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(selected), "gpu": gpu_name,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def to_device(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


@contextmanager
def record_training_failure(output, run):
    try:
        yield
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error={"type": type(error).__name__, "message": str(error)},
                   stopped_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "run.json", run)
        raise


@torch.no_grad()
def evaluate_loss(model, dataset, device, batch_size):
    """Weight each batch by valid action labels, including short episode tails."""
    previous_training = model.training
    model.eval()
    total, denominator = 0.0, 0
    try:
        for start in range(0, len(dataset), batch_size):
            batch = to_device(default_collate([dataset[index] for index in
                              range(start, min(start + batch_size, len(dataset)))]), device)
            labels = int(batch["mask"].sum().item()) * batch["action"].shape[-1]
            total += float(batch_loss(model, batch).item()) * labels
            denominator += labels
    finally:
        model.train(previous_training)
    if denominator < 1 or not np.isfinite(total):
        raise ValueError("Validation must contain finite, unmasked action labels")
    return total / denominator


def validate_training_manifest(data_dir, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Training requires a complete schema-version 1 dataset")
    if manifest.get("action_semantics") != "absolute_joint_position":
        raise ValueError("Airbot policy requires recorded absolute joint positions")
    if not manifest.get("depth_config") or not manifest.get("depth_provenance"):
        raise ValueError("Dataset lacks frozen depth preprocessing provenance")
    if (canonical_depth_config(manifest["depth_config"]) != manifest["depth_config"]
            or manifest["depth_config"]["image_size"] < 16):
        raise ValueError("Dataset depth preprocessing must be canonical with image_size >=16")
    if not manifest.get("camera_keys") or len(set(manifest["camera_keys"])) != len(manifest["camera_keys"]):
        raise ValueError("Dataset needs distinct ordered camera keys")
    for kind in ("state", "action"):
        dimension = manifest.get(f"{kind}_dim")
        names = manifest.get(f"{kind}_names", [])
        if type(dimension) is not int or dimension < 1 or len(names) != dimension:
            raise ValueError(f"Invalid {kind} dimensions or names")
    if not isinstance(manifest.get("prompt"), str) or not manifest["prompt"].strip():
        raise ValueError("Dataset needs a nonempty recorded prompt")
    if not np.isfinite(manifest.get("fps", float("nan"))) or manifest["fps"] <= 0:
        raise ValueError("Dataset frame rate must be finite and positive")
    train, validation = [select_episodes(manifest, split) for split in ("train", "validation")]
    if not train or not validation:
        raise ValueError("Training requires nonempty train and held-out validation episode splits")
    for key in ("episode_index", "path", "sha256"):
        if {item[key] for item in train} & {item[key] for item in validation}:
            raise ValueError(f"Train/validation episode leakage by {key}")
    root = Path(data_dir).resolve()
    for episode in train + validation:
        path = (root / episode["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Episode path escapes the converted data directory")
        if sha256_file(path) != episode["sha256"]:
            raise ValueError(f"Converted episode checksum mismatch: {episode['path']}")
    return train, validation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Completed converted dataset directory")
    parser.add_argument("--output", required=True, help="New or empty run directory; no resume/overwrite")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    for key in ("steps", "batch_size", "horizon", "checkpoint_every", "eval_every", "threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    output, data_dir = Path(args.output).resolve(), Path(args.data).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Refusing a nonempty run directory; choose a new --output")
    manifest_path = data_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = load_manifest(data_dir)
    training, validation = validate_training_manifest(data_dir, manifest)
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Dataset manifest changed during validation")
    statistics = fit_statistics(data_dir, training)
    vocabulary = build_vocabulary([{"instruction": manifest["prompt"]}])
    encode_language(manifest["prompt"], vocabulary)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    model_config = {"vocabulary_size": len(vocabulary), "state_dim": manifest["state_dim"],
                    "action_dim": manifest["action_dim"], "views": len(manifest["camera_keys"]),
                    "horizon": args.horizon}
    model = AirbotPolicyModel(**model_config).to(args.device)
    datasets = [AirbotDataset(data_dir, episodes, statistics, vocabulary, horizon=args.horizon)
                for episodes in (training, validation)]
    if any(len(dataset) < 1 for dataset in datasets):
        raise ValueError("Both dataset splits must contain policy observations")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(args.seed)
    config = vars(args).copy()
    config.update(data=str(data_dir), output=str(output))
    identity = {"manifest_sha256": manifest_sha256, "manifest": manifest,
                "source": source_identity(), "environment": environment_identity(args.device)}
    run = {"schema_version": 1, "format": CHECKPOINT_FORMAT, "status": "running",
           "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
           "model_config": model_config, "initialization": "all_policy_parameters_random",
           "image_input_precision": IMAGE_INPUT_PRECISION,
           "depth_encoder": "frozen_offline_depth_anything", "statistics": statistics,
           "vocabulary": vocabulary, "depth_config": manifest["depth_config"],
           "depth_provenance": manifest["depth_provenance"],
           "action_semantics": manifest["action_semantics"],
           "parameters": sum(parameter.numel() for parameter in model.parameters()),
           "train_episodes": len(training), "validation_episodes": len(validation),
           "train_frames": len(datasets[0]), "validation_frames": len(datasets[1]),
           "validation_metric": "masked_smooth_l1_on_normalized_absolute_actions",
           **identity}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run.json", run)
    print(json.dumps({key: run[key] for key in ("parameters", "train_episodes", "validation_episodes",
                                               "train_frames", "validation_frames", "model_config")}), flush=True)
    best, start_time = float("inf"), time.perf_counter()
    model.train()
    with record_training_failure(output, run), (output / "metrics.jsonl").open("x") as log:
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(datasets[0]), (args.batch_size,), generator=generator).tolist()
            batch = to_device(default_collate([datasets[0][index] for index in indices]), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "train_loss": float(loss.detach().item()),
                      "gradient_norm": float(gradient_norm.item()),
                      "elapsed_s": time.perf_counter() - start_time}
            retain = step % args.checkpoint_every == 0
            evaluate = step % args.eval_every == 0 or retain or step == args.steps
            if evaluate:
                validation_loss = evaluate_loss(model, datasets[1], args.device, args.batch_size)
                record["validation_loss"] = validation_loss
                improved = validation_loss < best
                best = min(best, validation_loss)
                payload = {"schema_version": 1, "format": CHECKPOINT_FORMAT, "config": config,
                           "image_input_precision": IMAGE_INPUT_PRECISION,
                           "model_config": model_config, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "step": step,
                           "validation_loss": validation_loss, "best_validation_loss": best,
                           "statistics": statistics, "vocabulary": vocabulary,
                           "depth_config": manifest["depth_config"],
                           "depth_provenance": manifest["depth_provenance"],
                           "action_semantics": manifest["action_semantics"], **identity,
                           "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(args.device).type == "cuda" else None}
                if retain:
                    destination = output / f"step_{step:06d}.pt"
                    if destination.exists():
                        raise FileExistsError(f"Refusing to overwrite retained checkpoint: {destination}")
                    save_checkpoint(destination, payload)
                save_checkpoint(output / "latest.pt", payload)
                if improved:
                    save_checkpoint(output / "best.pt", payload)
            if evaluate or step % 25 == 0 or step == 1:
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
    run.update(status="complete", completed_steps=args.steps, best_validation_loss=best,
               elapsed_s=time.perf_counter() - start_time)
    atomic_json(output / "run.json", run)


if __name__ == "__main__":
    main()
