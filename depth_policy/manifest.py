import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from depth_policy.common import atomic_json, load_config, sha256_file
from depth_policy.simulation import task_and_states


def create_manifest(states, config, initial_file):
    hashes = [hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() for state in states]
    unique = list(dict.fromkeys(hashes))
    if len(unique) < 10:
        raise ValueError("Too few independent initial states for the pilot")
    order = np.random.default_rng(config["seed"]).permutation(len(unique))
    train_end = int(0.7 * len(unique))
    validation_end = int(0.85 * len(unique))
    assignment = {}
    for rank, index in enumerate(order):
        assignment[unique[index]] = "train" if rank < train_end else (
            "validation" if rank < validation_end else "test"
        )
    return {
        "schema_version": 1, "task": config["task"], "suite": config["suite"],
        "seed": config["seed"], "initial_file_sha256": sha256_file(initial_file),
        "total_states": len(states), "unique_states": len(unique),
        "states": [{"id": index, "sha256": digest, "split": assignment[digest]}
                   for index, digest in enumerate(hashes)],
        "state_std": np.std(states, axis=0).tolist(),
    }


def load_manifest(config, states, initial_file):
    manifest = json.loads(Path(config["manifest"]).read_text())
    expected = create_manifest(states, config, initial_file)
    if manifest != expected:
        raise ValueError("Manifest does not match the task, initial states, or split configuration")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    args = parser.parse_args()
    config = load_config(args.config)
    _, states, path = task_and_states(config)
    if Path(config["manifest"]).exists():
        manifest = load_manifest(config, states, path)
    else:
        manifest = create_manifest(states, config, path)
        atomic_json(config["manifest"], manifest)
    counts = {split: sum(item["split"] == split for item in manifest["states"])
              for split in ("train", "validation", "test")}
    print(json.dumps({"total": len(states), "unique": manifest["unique_states"], "counts": counts}))


if __name__ == "__main__":
    main()
