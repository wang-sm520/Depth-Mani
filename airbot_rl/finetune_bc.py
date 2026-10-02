"""Train the wrist-depth BC on replay datasets (replay.py), initialised from an existing checkpoint.

Every domain contributes the same number of frames per step; the loss is their weighted sum. The saved
checkpoint keeps the base's format, statistics and head DA2 config (AirbotDepthPolicy loads it unchanged) and
adds "wrist_depth" so deployment knows the wrist input is D405 metric depth.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from airbot_depth.data import AirbotDataset
from airbot_depth.model import batch_loss
from airbot_depth.policy import AirbotDepthPolicy
from airbot_depth.train import evaluate_loss, save_checkpoint, seed_everything, to_device

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="initial weights")
parser.add_argument("--data", nargs="+", required=True, help="replay dataset directories (e.g. .../mixed .../sim)")
parser.add_argument("--weights", type=float, nargs="+", required=True, help="loss weight per --data")
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--batch", type=int, required=True, help="frames per domain per step")
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
assert len(args.data) == len(args.weights)

seed_everything(args.seed)
base = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = AirbotDepthPolicy(args.checkpoint).model.to(args.device).train().requires_grad_(True)
domains = {}
for root in map(Path, args.data):
    manifest = json.loads((root / "manifest.json").read_text())
    domains[root.name] = {split: AirbotDataset(root, [e for e in manifest["episodes"] if e["split"] == split],
                                               base["statistics"], base["vocabulary"], base["config"]["horizon"])
                          for split in ("train", "validation")}
    wrist = manifest["wrist"]
weights = dict(zip(domains, args.weights))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
generator = torch.Generator().manual_seed(args.seed)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
(out / "args.json").write_text(json.dumps(vars(args), indent=1))
best = float("inf")
with (out / "metrics.jsonl").open("w") as log:
    for step in range(args.steps + 1):
        record = {"step": step}
        if step:
            optimizer.zero_grad(set_to_none=True)
            for name, splits in domains.items():
                index = torch.randint(len(splits["train"]), (args.batch,), generator=generator).tolist()
                loss = batch_loss(model, to_device(default_collate([splits["train"][i] for i in index]), args.device))
                (weights[name] * loss).backward()
                record[f"train_{name}"] = loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            for name, splits in domains.items():
                record[f"val_{name}"] = evaluate_loss(model, splits["validation"], args.device, 256)
            record["val"] = sum(weights[n] * record[f"val_{n}"] for n in domains)
            payload = {**base, "model": model.state_dict(), "step": step, "validation_loss": record["val"],
                       "wrist_depth": wrist, "finetune": vars(args)}
            save_checkpoint(out / f"step_{step:06d}.pt", payload)
            if record["val"] < best:
                best = record["val"]
                save_checkpoint(out / "best.pt", payload)
            print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
