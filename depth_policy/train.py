import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from depth_policy.common import ROOT, atomic_json, source_identity
from depth_policy.data import (
    EpisodeDataset, build_vocabulary, data_fingerprint, fit_statistics, training_episodes,
)
from depth_policy.model import StudentPolicy, masked_action_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_snapshot(output, payload, interval):
    if interval and payload["step"] % interval == 0:
        destination = output / f"step_{payload['step']:06d}.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite retained checkpoint: {destination}")
        save_checkpoint(destination, payload)


def to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def batch_loss(model, batch):
    prediction = model(batch["state"], batch["tokens"], batch.get("images"))
    return masked_action_loss(prediction, batch["action"], batch["mask"])


@torch.no_grad()
def evaluate_loss(model, dataset, device, batch_size):
    model.eval()
    total, denominator = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        batch = to_device(default_collate([dataset[index] for index in
                          range(start, min(len(dataset), start + batch_size))]), device)
        valid = int(batch["mask"].sum())
        total += float(batch_loss(model, batch)) * valid
        denominator += valid
    model.train()
    return total / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/teacher")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modality", choices=["depth", "rgb", "state"], default="depth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.save_every < 1 or args.checkpoint_every < 0 or args.checkpoint_every % args.save_every:
        raise ValueError("checkpoint-every must be zero or a positive multiple of save-every")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty run directory; use a new output or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    training = training_episodes(args.data, "train", args.limit)
    validation = training_episodes(args.data, "validation")
    if {item["initial_state_sha256"] for item in training} & {
        item["initial_state_sha256"] for item in validation
    }:
        raise ValueError("Initial-state leakage between training and validation")
    fingerprint = {"train": data_fingerprint(training), "validation": data_fingerprint(validation)}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint["data"] != fingerprint:
            raise ValueError("Resume data differs from the checkpoint")
        for name in ("modality", "batch_size", "learning_rate", "seed", "horizon", "image_size", "depth_max", "limit"):
            if checkpoint["config"][name] != getattr(args, name):
                raise ValueError(f"Resume configuration mismatch: {name}")
    statistics = checkpoint["statistics"] if checkpoint else fit_statistics(training)
    vocabulary = checkpoint["vocabulary"] if checkpoint else build_vocabulary(training)
    datasets = [EpisodeDataset(episodes, args.modality, vocabulary, statistics, args.horizon,
                              args.image_size, args.depth_max) for episodes in (training, validation)]
    model = StudentPolicy(args.modality, len(vocabulary), args.horizon).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(args.seed)
    first_step, best = 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step, best = checkpoint["step"], checkpoint["best_validation_loss"]
        generator.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if args.device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    config = vars(args)
    summary = {"config": config, "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "train_frames": len(datasets[0]), "validation_frames": len(datasets[1]),
               "train_episodes": len(training), "validation_episodes": len(validation),
               "train_initial_states": len({item["initial_state_id"] for item in training}),
               "initialization": "all_student_parameters_random", "statistics": statistics,
               "vocabulary": vocabulary, "data": fingerprint,
               "source": source_identity(ROOT / "depth_policy")}
    atomic_json(output / "run.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"data", "statistics", "source"}}), flush=True)
    start_time = time.perf_counter()
    model.train()
    with (output / "metrics.jsonl").open("a") as log:
        for step in range(first_step, args.steps):
            indices = torch.randint(len(datasets[0]), (args.batch_size,), generator=generator).tolist()
            batch = to_device(default_collate([datasets[0][index] for index in indices]), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record = {"step": step + 1, "train_loss": float(loss.detach()),
                      "gradient_norm": float(gradient_norm), "elapsed_s": time.perf_counter() - start_time}
            should_save = (step + 1) % args.save_every == 0 or step + 1 == args.steps
            if should_save:
                validation_loss = evaluate_loss(model, datasets[1], args.device, args.batch_size)
                record["validation_loss"] = validation_loss
                improved = validation_loss < best
                best = min(best, validation_loss)
                payload = {"config": config, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "step": step + 1,
                           "best_validation_loss": best, "statistics": statistics,
                           "vocabulary": vocabulary, "data": fingerprint,
                           "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None}
                save_snapshot(output, payload, args.checkpoint_every)
                save_checkpoint(output / "latest.pt", payload)
                if improved:
                    save_checkpoint(output / "best.pt", payload)
            if should_save or (step + 1) % 25 == 0 or step == first_step:
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
