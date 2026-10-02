import argparse
import gc
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from depth_policy.common import ROOT, atomic_json, source_identity
from depth_policy.data import EpisodeDataset, build_vocabulary, data_fingerprint, fit_statistics, training_episodes
from depth_policy.model import StudentPolicy
from depth_policy.train import batch_loss, seed_everything, to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/teacher")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="reports/student-cpu-budget.json")
    args = parser.parse_args()
    if args.steps < 5 or args.limit < 1 or args.batch_size < 1:
        raise ValueError("Require >=5 steps, >=1 episode and a positive batch size")
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this CPU budget test with CUDA_VISIBLE_DEVICES=''")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if Path(args.output).exists():
        raise FileExistsError("Choose a new report path instead of replacing prior measurements")
    torch.set_num_threads(args.threads)
    episodes = training_episodes(args.data, "train", args.limit)
    vocabulary = build_vocabulary(episodes)
    statistics = fit_statistics(episodes)
    report = {
        "purpose": "Resource preflight only; no checkpoint or formal experiment result produced",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "config": vars(args), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source": source_identity(ROOT / "depth_policy"), "data": data_fingerprint(episodes),
        "data_split": "train", "validation_or_test_used": False, "results": {}, "complete": False,
    }
    atomic_json(args.output, report)
    for modality in ["depth", "rgb", "state"]:
        seed_everything(0)
        start = time.perf_counter()
        dataset = EpisodeDataset(episodes, modality, vocabulary, statistics)
        load_seconds = time.perf_counter() - start
        model = StudentPolicy(modality, len(vocabulary)).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        generator = torch.Generator().manual_seed(0)
        timings, losses = [], []
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        for step in range(args.steps):
            start = time.perf_counter()
            indices = torch.randint(len(dataset), (args.batch_size,), generator=generator).tolist()
            batch = to_device(default_collate([dataset[index] for index in indices]), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite preflight loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
        steady = np.asarray(timings[3:])
        with torch.no_grad():
            model.eval()
            example = to_device(default_collate([dataset[0]]), args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            inference_timings = []
            for repeat in range(10):
                start = time.perf_counter()
                model(example["state"], example["tokens"], example.get("images"))
                if args.device == "cuda":
                    torch.cuda.synchronize()
                inference_timings.append(time.perf_counter() - start)
        result = {
            "peak_cuda_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "data_frames": len(dataset), "load_seconds": load_seconds,
            "steps": args.steps, "step_seconds": timings,
            "steady_step_median_seconds": float(np.median(steady)),
            "steady_step_p95_seconds": float(np.percentile(steady, 95)),
            "estimated_10000_steps_hours_excluding_validation": float(steady.mean() * 10000 / 3600),
            "forward_only_batch1_median_seconds": float(np.median(inference_timings[1:])),
            "peak_process_rss_GiB_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "first_loss_diagnostic_only": losses[0], "last_loss_diagnostic_only": losses[-1],
        }
        report["results"][modality] = result
        atomic_json(args.output, report)
        print(f"{modality}: {result}", flush=True)
        del dataset, model, optimizer, batch, example, loss
        gc.collect()
    report["complete"] = True
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
