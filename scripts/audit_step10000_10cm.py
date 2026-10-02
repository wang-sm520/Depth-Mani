import collections
import hashlib
import json

import h5py
import matplotlib
import numpy as np
import torch

from depth_policy.common import ROOT, atomic_json, sha256_file
from depth_policy.episodes import episode_index
from depth_policy.evaluate import Student
from depth_policy.evaluate_suite import check_records
from depth_policy.validate import validate_episode

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    torch.set_num_threads(4)
    bank_path = ROOT / "data/wine/evaluation/random100-10cm.json"
    bank = json.loads(bank_path.read_text())
    digest = sha256_file(bank_path)
    result = {"bank_sha256": digest, "training_step": 10000, "groups": {}}
    checkpoints = {}
    outcomes = {}
    for modality in ("depth", "rgb"):
        checkpoint_path = ROOT / f"runs/put_the_wine_bottle_on_top_of_the_cabinet/{modality}-300-seed0/latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 10000 and checkpoint["config"]["limit"] == 300
        checkpoints[modality] = checkpoint
        policy = Student(checkpoint_path, "cpu")
        assert policy.config["modality"] == modality
        directory = ROOT / f"runs/wine300-evaluation/{modality}-step10000-random100-10cm"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["completed"] and summary["policy"] == policy.metadata
        assert summary["evaluation_bank_sha256"] == digest
        assert not list(directory.glob("*.partial.h5"))
        records = episode_index(directory)
        assert len(check_records(records, bank, policy.metadata, digest, 4)) == 100
        indexed = {record["initial_state_id"]: record for record in records}
        audits = []
        for entry in bank["states"]:
            record = indexed[entry["id"]]
            assert record["config"] == {**bank["config"], "teacher_replan_steps": 4}
            with h5py.File(record["path"], "r") as stream:
                assert not stream.attrs["error"]
                np.testing.assert_array_equal(stream["initial_sim_state"][:], entry["state"])
                assert hashlib.sha256(stream["model_xml"].asstr()[()].encode()).hexdigest() == entry["xml_sha256"]
                steps = stream["steps"]
                assert len(steps["phase"]) <= 310
                if not record["success"]:
                    assert len(steps["phase"]) == 310
                first = int(np.flatnonzero(steps["phase"][:] == 1)[0])
                observation = {name: steps[name][first] for name in
                               ("state", f"{modality}_agentview", f"{modality}_wrist")}
                actions = policy.infer(observation, record["instruction"])
                count = min(4, len(steps["phase"]) - first)
                np.testing.assert_allclose(actions[:count], steps["executed_action"][first:first + count],
                                           atol=1e-6, rtol=0)
            audits.append(validate_episode(record["path"], bank))
        success = np.asarray([indexed[entry["id"]]["success"] for entry in bank["states"]])
        outcomes[modality] = success
        assert int(success.sum()) == summary["successes"]
        positions = np.asarray([entry["initial_positions"]["wine_bottle_1"][:2]
                                for entry in bank["states"]])
        offsets = (positions - [-0.20, -0.05]) * 100
        figure, axis = plt.subplots(figsize=(8, 8))
        for mask, color, marker, label in (
                (success, "#218c54", "o", f"Success: {success.sum()}/100"),
                (~success, "#d24747", "X", f"Timeout: {(~success).sum()}/100")):
            axis.scatter(offsets[mask, 0], offsets[mask, 1], c=color, marker=marker,
                         s=65, label=label, edgecolors="white", linewidths=0.5)
        axis.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, linestyle="--",
                                 edgecolor="#496a9a", linewidth=1.5, label="Original 3 x 3 cm center range"))
        axis.set(xlim=(-5.3, 5.3), ylim=(-5.3, 5.3), aspect="equal",
                 xlabel="Bottle initial x offset from nominal center (cm)",
                 ylabel="Bottle initial y offset from nominal center (cm)",
                 title=f"{modality.upper()} | 300 demos | checkpoint step 10,000\n"
                       f"10 x 10 cm placement test: {success.sum()}/100 ({success.mean():.0%})")
        axis.set_xticks(np.arange(-5, 6))
        axis.set_yticks(np.arange(-5, 6))
        axis.grid(alpha=0.18)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
        figure.tight_layout()
        destination = ROOT / f"reports/figures/wine300-{modality}-step10000-10cm-success-map.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=170, bbox_inches="tight")
        plt.close(figure)
        result["groups"][modality] = {
            "policy": policy.metadata, "attempts": 100, "successes": int(success.sum()),
            "success_rate": float(success.mean()), "figure": str(destination),
            "termination_counts": dict(collections.Counter(record["termination_reason"] for record in records)),
            "first_action_chunks_verified": True, "trajectory_audit": audits,
        }
        print(json.dumps({"modality": modality, "successes": int(success.sum())}), flush=True)
    for key in ("data", "statistics", "vocabulary"):
        assert checkpoints["depth"][key] == checkpoints["rgb"][key]
    result["same_training_data_statistics_vocabulary"] = True
    result["paired"] = {
        "both_success": int((outcomes["depth"] & outcomes["rgb"]).sum()),
        "depth_only_success": int((outcomes["depth"] & ~outcomes["rgb"]).sum()),
        "rgb_only_success": int((~outcomes["depth"] & outcomes["rgb"]).sum()),
        "both_fail": int((~outcomes["depth"] & ~outcomes["rgb"]).sum()),
    }
    atomic_json(ROOT / "reports/wine300-step10000-10cm-audit.json", result)


if __name__ == "__main__":
    main()
