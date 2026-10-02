import collections
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from depth_policy.common import ROOT, atomic_json, sha256_file
from depth_policy.data import data_fingerprint, training_episodes
from depth_policy.episodes import episode_index
from depth_policy.evaluate import Student
from depth_policy.evaluate_suite import check_records
from depth_policy.validate import validate_episode


def audit_group(group, policies):
    bank_path = ROOT / f"data/wine/evaluation/{group}.json"
    bank = json.loads(bank_path.read_text())
    digest = sha256_file(bank_path)
    expected_config = {**bank["config"], "teacher_replan_steps": 4}
    records_by_modality = {}
    summaries = {}
    audits = []
    for modality, policy in policies.items():
        directory = ROOT / ("runs/wine300-evaluation" if modality == "depth"
                            else "runs/wine300-rgb-comparison") / group
        summary = json.loads((directory / "summary.json").read_text())
        records = episode_index(directory)
        assert not list(directory.glob("*.partial.h5"))
        assert summary["completed"] and summary["completed_attempts"] == 100
        assert summary["evaluation_bank_sha256"] == digest
        assert summary["policy"] == policy.metadata
        assert len(check_records(records, bank, policy.metadata, digest, 4)) == 100
        summaries[modality] = summary
        records_by_modality[modality] = {record["initial_state_id"]: record for record in records}
        for record in records:
            assert record["config"] == expected_config
            entry = bank["states"][record["initial_state_id"]]
            with h5py.File(record["path"], "r") as stream:
                np.testing.assert_allclose(stream["initial_sim_state"][:], entry["state"],
                                           atol=1e-12, rtol=0)
                xml_digest = hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest()
                assert xml_digest == entry["xml_sha256"]
                assert not stream.attrs["error"]
                steps = stream["steps"]
                assert len(steps["phase"]) <= 310
                if not record["success"]:
                    assert len(steps["phase"]) == 310
                if modality == "rgb":
                    first = int(np.flatnonzero(steps["phase"][:] == 1)[0])
                    observation = {name: steps[name][first] for name in
                                   ("state", "rgb_agentview", "rgb_wrist")}
                    actions = policy.infer(observation, record["instruction"])
                    count = min(4, len(steps["phase"]) - first)
                    np.testing.assert_allclose(actions[:count],
                                               steps["executed_action"][first:first + count],
                                               atol=1e-6, rtol=0)
            if modality == "rgb":
                audits.append(validate_episode(record["path"], bank))
    paired = []
    for entry in bank["states"]:
        depth = records_by_modality["depth"][entry["id"]]
        rgb = records_by_modality["rgb"][entry["id"]]
        for key in ("camera", "controller_config", "instruction", "config", "seed"):
            assert depth[key] == rgb[key]
        paired.append({"initial_state_id": entry["id"], "seed": entry["seed"],
                       "depth_success": depth["success"], "rgb_success": rgb["success"],
                       "depth_path": depth["path"], "rgb_path": rgb["path"]})
    counts = {modality: sum(row[f"{modality}_success"] for row in paired)
              for modality in policies}
    for modality, count in counts.items():
        assert count == summaries[modality]["successes"]
    result = {
        "bank_sha256": digest, "attempts_per_policy": len(paired), "successes": counts,
        "rgb_minus_depth_percentage_points": counts["rgb"] - counts["depth"],
        "both_success": sum(row["depth_success"] and row["rgb_success"] for row in paired),
        "depth_only_success": sum(row["depth_success"] and not row["rgb_success"] for row in paired),
        "rgb_only_success": sum(row["rgb_success"] and not row["depth_success"] for row in paired),
        "both_fail": sum(not row["depth_success"] and not row["rgb_success"] for row in paired),
        "termination_counts": {modality: dict(collections.Counter(record["termination_reason"]
                               for record in records.values()))
                               for modality, records in records_by_modality.items()},
        "paired_episodes": paired, "rgb_trajectory_audit": audits,
        "latency_note": "RGB groups ran concurrently on disjoint CPU sets; not a controlled latency benchmark.",
    }
    print(json.dumps({"group": group, "successes": counts}), flush=True)
    return result


def main():
    torch.set_num_threads(4)
    root = ROOT / "runs/put_the_wine_bottle_on_top_of_the_cabinet"
    runs = {modality: json.loads((root / f"{modality}-300-seed0/run.json").read_text())
            for modality in ("depth", "rgb")}
    policies = {modality: Student(root / f"{modality}-300-seed0/best.pt", "cpu")
                for modality in runs}
    for key in runs["depth"]["config"]:
        if key not in ("output", "modality"):
            assert runs["depth"]["config"][key] == runs["rgb"]["config"][key]
    for key in ("data", "statistics", "vocabulary", "train_frames", "source"):
        assert runs["depth"][key] == runs["rgb"][key]
    for split in ("train", "validation"):
        episodes = training_episodes(ROOT / "data/wine/teacher", split, 300 if split == "train" else None)
        assert len(episodes) == (300 if split == "train" else 7)
        assert data_fingerprint(episodes) == runs["rgb"]["data"][split]
    for modality, policy in policies.items():
        checkpoint = torch.load(root / f"{modality}-300-seed0/best.pt", map_location="cpu", weights_only=False)
        for key in ("config", "data", "statistics", "vocabulary"):
            assert checkpoint[key] == runs[modality][key]
        latest = torch.load(root / f"{modality}-300-seed0/latest.pt", map_location="cpu", weights_only=False)
        assert latest["step"] == 10000
        assert policy.config["modality"] == modality
    results = {group: audit_group(group, policies) for group in ("random100", "random100-10cm")}
    atomic_json(ROOT / "reports/wine300-rgb-comparison-audit.json", {
        "training_data_unchanged": True, "training_config_matches_except_modality_and_output": True,
        "policies": {modality: policy.metadata for modality, policy in policies.items()},
        "groups": results,
    })


if __name__ == "__main__":
    main()
