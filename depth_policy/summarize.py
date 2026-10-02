import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np

from depth_policy.common import atomic_json
from depth_policy.episodes import episode_index


def latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"queries": 0}
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid inference timings")
    return {"queries": len(values), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)), "max_seconds": float(values.max()),
            "total_seconds": float(values.sum())}


def summarize(directory):
    directory = Path(directory)
    episodes = episode_index(directory)
    incomplete = sorted(str(path) for path in directory.glob("*.partial.h5"))
    grouped = collections.defaultdict(list)
    for episode in episodes:
        grouped[episode["split"]].append(episode)
    splits = {}
    all_times = []
    total_frames = 0
    total_bytes = 0
    for split, records in sorted(grouped.items()):
        times = []
        frames = 0
        state_results = collections.defaultdict(list)
        episode_results = []
        for episode in records:
            with h5py.File(episode["path"], "r") as stream:
                timings = stream["inference_seconds"][:]
                demonstration_frames = int(np.sum(stream["steps/phase"][:] == 1)) if "phase" in stream["steps"] else 0
                episode_frames = len(stream["steps/state"]) if "state" in stream["steps"] else 0
                times.extend(timings.tolist())
                frames += episode_frames
            total_bytes += Path(episode["path"]).stat().st_size
            state_results[episode["initial_state_id"]].append(episode["success"])
            episode_results.append({"episode_id": episode["episode_id"], "path": episode["path"],
                                    "initial_state_id": episode["initial_state_id"], "seed": episode["seed"],
                                    "success": episode["success"], "termination_reason": episode["termination_reason"],
                                    "frames": episode_frames, "demonstration_frames": demonstration_frames,
                                    "inference": latency_summary(timings)})
        successes = sum(episode["success"] for episode in records)
        splits[split] = {
            "attempts": len(records), "successes": successes, "success_rate": successes / len(records),
            "unique_initial_states": len(state_results), "frames": frames,
            "termination_counts": dict(collections.Counter(episode["termination_reason"] for episode in records)),
            "per_initial_state": {str(key): {"attempts": len(values), "successes": sum(values)}
                                  for key, values in sorted(state_results.items())},
            "inference": latency_summary(times), "episodes": episode_results,
        }
        all_times.extend(times)
        total_frames += frames
    successes = sum(episode["success"] for episode in episodes)
    total_attempts = len(episodes) + len(incomplete)
    identities = {json.dumps(episode["teacher"], sort_keys=True) for episode in episodes}
    return {
        "directory": str(directory), "completed_attempts": len(episodes),
        "incomplete_attempts": len(incomplete), "incomplete_files": incomplete,
        "total_recorded_attempts": total_attempts, "successes": successes,
        "success_rate_all_recorded_attempts": successes / total_attempts if total_attempts else None,
        "frames": total_frames, "completed_episode_bytes": total_bytes,
        "policies": [json.loads(identity) for identity in sorted(identities)],
        "inference": latency_summary(all_times), "splits": splits,
        "latency_definition": "Wall time around policy.infer; includes preprocessing and client transport for teacher; compilation is not excluded.",
        "scope": "Only recorded attempts; setup failures before opening an episode must be reported from runtime logs separately.",
        "independence_note": "Repeated rollouts from the same initial state are not additional independent scenes.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.data)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ["completed_attempts", "incomplete_attempts", "successes", "inference"]}))


if __name__ == "__main__":
    main()
