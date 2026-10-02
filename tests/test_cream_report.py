import json

import numpy as np
import pytest
from matplotlib.axes import Axes

from depth_policy.common import ROOT, atomic_json
from depth_policy.evaluate_suite import state_hash
import scripts.cream_cheese50k_report as report


def pose(x, y, yaw=0.0):
    return {"position_xyz_m": [x, y, 0.9],
            "quaternion_wxyz": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
            "yaw_rad": yaw}


def attempt(identifier, split, status="success"):
    cheese = pose(-0.06 + identifier * 0.005, 0.12, identifier / 10)
    bowl = pose(-0.10, 0.11)
    state = np.asarray([identifier + 0.1, 0.4], dtype=np.float64)
    return {"id": identifier, "split": split, "seed": 100 + identifier, "status": status,
            "initial_state_sha256": state_hash(state), "initial_sim_state": state.tolist(),
            "xml_sha256": "a" * 64, "object_initial_poses": {"cream_cheese_1": cheese,
                                                           "akita_black_bowl_1": bowl},
            "relative_cream_cheese_to_bowl_xyz_m": [cheese["position_xyz_m"][0] + 0.10, 0.01, 0],
            "path": f"episode-{identifier}.h5", "error": None}


def schedule_and_ledger():
    states = [{"id": index, "seed": 100 + index, "split": "train" if index < 3 else "validation"}
              for index in range(5)]
    schedule = {"schema_version": 1, "states": states, "official_initial_state_sha256": []}
    ledger = {"schema_version": 1, "entries": [attempt(0, "train"), attempt(1, "train", "timeout"),
                                                attempt(3, "validation")]}
    return schedule, ledger


def test_protocol_has_3000_frozen_paired_trials():
    config = json.loads((ROOT / "configs/cream_cheese50k_on_experiment.json").read_text())
    assert config["official_instruction"] == "Put the cream cheese on the bowl"
    assert config["attempt_bank"] == "data/cream_cheese/attempts-on.json"
    assert config["training_data"] == "data/cream_cheese/teacher-on"
    assert config["training_limit"] == 300 and config["validation_limit"] == 7
    assert config["training_steps"] == 50000
    assert config["checkpoint_steps"] == [10000, 20000, 30000, 40000, 50000]
    assert {group: settings["count"] for group, settings in config["groups"].items()} == {
        "official-range100": 100, "expanded3x200": 200}
    assert len(config["modalities"]) * len(config["checkpoint_steps"]) * sum(
        item["count"] for item in config["groups"].values()) == 3000


def test_official_bddl_instruction_rejects_suite_derived_instruction():
    config = {"official_instruction": "Put the cream cheese on the bowl"}
    report.require_official_instruction("Put the cream cheese on the bowl", config)
    with pytest.raises(ValueError, match="official BDDL instruction"):
        report.require_official_instruction("Put the cream cheese in the bowl", config)


def test_collection_attempts_preserve_failures_and_seed_schedule():
    schedule, ledger = schedule_and_ledger()
    indexed = report.validate_attempt_ledger(schedule, ledger)
    assert set(indexed) == {0, 1, 3}
    assert indexed[1]["status"] == "timeout"
    result = report.coverage_summary([indexed[0], indexed[1]])
    assert result["cream_cheese_1"]["attempted_occupied_cells"] == 2
    assert result["cream_cheese_1"]["successful_occupied_cells"] == 1
    assert result["cheese_minus_bowl"]["attempted_xy_min_m"][0] == pytest.approx(0.04)
    assert "successful_yaw_range_rad" in result["akita_black_bowl_1"]


def test_collector_source_audit_only_allows_documented_replay_change():
    original = {"collect_random.py": "collector", "simulation.py": "scene", "replay.py": "old"}
    upgraded = {**original, "replay.py": "rgb-verified"}
    records = [{"split": "train", "initial_state_id": 0, "collector_source": original},
               {"split": "validation", "initial_state_id": 3000, "collector_source": upgraded}]
    audit = report.audit_collector_sources(records, upgraded)
    assert len(audit["versions"]) == 2
    assert audit["current_changes_from_training_baseline"] == ["replay.py"]
    assert {version["attempt_ids"][0] for version in audit["versions"]} == {0, 3000}
    with pytest.raises(ValueError, match="Collection implementation changed"):
        report.audit_collector_sources(records, {**upgraded, "simulation.py": "changed"})
    with pytest.raises(ValueError, match="Collector source file list changed"):
        report.audit_collector_sources(records, {"replay.py": "rgb-verified"})


@pytest.mark.parametrize("change", ["reused_seed", "reused_state", "skipped_seed", "wrong_relative",
                                    "bad_yaw", "yaw_disagrees_with_quaternion"])
def test_collection_ledger_rejects_leaks_and_pose_errors(change):
    schedule, ledger = schedule_and_ledger()
    if change == "reused_seed":
        schedule["states"][4]["seed"] = 100
    elif change == "reused_state":
        ledger["entries"][1]["initial_state_sha256"] = ledger["entries"][0]["initial_state_sha256"]
    elif change == "skipped_seed":
        ledger["entries"][1] = attempt(2, "train")
    elif change == "wrong_relative":
        ledger["entries"][0]["relative_cream_cheese_to_bowl_xyz_m"] = [0, 0, 0]
    elif change == "bad_yaw":
        ledger["entries"][0]["object_initial_poses"]["akita_black_bowl_1"]["yaw_rad"] = float("nan")
    else:
        ledger["entries"][0]["object_initial_poses"]["akita_black_bowl_1"]["yaw_rad"] = 1.0
    with pytest.raises((ValueError, AssertionError)):
        report.validate_attempt_ledger(schedule, ledger)


def test_training_metrics_require_every_500_step_validation(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps({"step": step, "train_loss": 0.1,
                                               "validation_loss": 0.2}) for step in (500, 1000)))
    assert len(report.training_metrics(tmp_path, 1000)) == 2
    metrics.write_text(json.dumps({"step": 1000, "train_loss": 0.1, "validation_loss": 0.2}))
    with pytest.raises(ValueError, match="validation"):
        report.training_metrics(tmp_path, 1000)


def test_evaluation_bank_checks_isolation_and_exact_tripled_width(tmp_path, monkeypatch):
    config = {"attempt_bank": "data/cream_cheese/attempts-on.json", "label": "test",
              "official_instruction": "Put the cream cheese on the bowl",
              "groups": {"official-range100": {"config": "configs/official.json", "seed": 1000, "count": 100},
                         "expanded3x200": {"config": "configs/expanded.json", "seed": 2000, "count": 200}}}
    original = {"minimum_xy_m": [-0.07, 0.11], "maximum_xy_m": [-0.03, 0.15]}
    expanded = {"minimum_xy_m": [-0.11, 0.07], "maximum_xy_m": [0.01, 0.19]}
    atomic_json(tmp_path / config["attempt_bank"], {"states": [{"seed": 100}]})
    monkeypatch.setattr(report, "ROOT", tmp_path)
    for name, settings in config["groups"].items():
        scene = {"official_instruction": config["official_instruction"],
                 "official_center_bounds": {report.CHEESE: original}}
        if name == "expanded3x200":
            scene["evaluation_center_bounds"] = {report.CHEESE: expanded}
        atomic_json(tmp_path / settings["config"], scene)
        states = []
        for index in range(settings["count"]):
            state = np.asarray([settings["seed"] + index], dtype=np.float64)
            states.append({"id": index, "seed": settings["seed"] + index, "sha256": state_hash(state),
                           "state": state.tolist(), "split": "random-test", "reset_and_settle_reproduced": True,
                           "xml_sha256": "a" * 64,
                           "initial_positions": {report.CHEESE: [-0.05, 0.13, 0.9],
                                                 report.BOWL: [-0.09, 0.11, 0.9]},
                           "initial_quaternions": {report.CHEESE: [1, 0, 0, 0],
                                                   report.BOWL: [1, 0, 0, 0]}})
        atomic_json(tmp_path / "data/cream_cheese/evaluation/test" / f"{name}.json",
                    {"mode": "random", "seed": settings["seed"], "config": scene, "states": states})
    collection = {"all_sampled_initial_state_sha256": [state_hash(np.asarray([100.], dtype=np.float64))]}
    assert len(report.validate_evaluation_banks(config, collection)) == 2
    path = tmp_path / "data/cream_cheese/evaluation/test/official-range100.json"
    bank = json.loads(path.read_text())
    bank["states"][0]["sha256"] = collection["all_sampled_initial_state_sha256"][0]
    atomic_json(path, bank)
    with pytest.raises(ValueError, match="Duplicated, leaked"):
        report.validate_evaluation_banks(config, collection)
    bank["states"][0]["sha256"] = state_hash(np.asarray([1000.], dtype=np.float64))
    bank["config"]["official_instruction"] = "Put the cream cheese in the bowl"
    atomic_json(path, bank)
    with pytest.raises(ValueError, match="Evaluation bank identity"):
        report.validate_evaluation_banks(config, collection)


def test_figures_include_coverage_relative_bowl_and_losses(tmp_path, monkeypatch):
    config = {"label": "test", "training_steps": 1000, "modalities": ["rgb", "depth"],
              "checkpoint_steps": [10000, 20000, 30000, 40000, 50000],
              "groups": {"official-range100": {"count": 100}, "expanded3x200": {"count": 200}}}
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "training_metrics", lambda directory, steps: [
        {"step": 500, "train_loss": 0.4, "validation_loss": 0.45},
        {"step": 1000, "train_loss": 0.3, "validation_loss": 0.39}])
    a, b = attempt(0, "train"), attempt(1, "train", "timeout")
    c = attempt(2, "validation")
    attempts = {0: a, 1: b, 2: c}
    coverage = {split: report.coverage_summary([entry for entry in attempts.values() if entry["split"] == split])
                for split in ("train", "validation")}
    banks, rows, details = {}, [], {}
    original = {"minimum_xy_m": [-0.07, 0.11], "maximum_xy_m": [-0.03, 0.15]}
    for group, settings in config["groups"].items():
        states = [{"initial_positions": {report.CHEESE: [-0.05 + index * 0.0001, 0.12, 0.9],
                                         report.BOWL: [-0.09, 0.11 + index * 0.0001, 0.9]}}
                  for index in range(settings["count"])]
        banks[group] = {"bank": {"states": states, "config": {"official_center_bounds": {report.CHEESE: original}}}}
        for modality in config["modalities"]:
            for step in config["checkpoint_steps"]:
                rows.append({"distribution": group, "modality": modality, "training_step": step,
                             "successes": settings["count"] // 2, "success_rate": 0.5})
                details[f"{modality}-{step}-{group}"] = {
                    "success_by_initial_state": (np.arange(settings["count"]) % 2 == 0).tolist()}
    histogram_bins = []
    original_hist = Axes.hist

    def capture_histogram(self, values, *args, **kwargs):
        histogram_bins.append(np.asarray(kwargs["bins"]))
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture_histogram)
    destination = report.make_figures(config, rows, details, banks, {"coverage": coverage}, attempts, tmp_path)
    assert histogram_bins and len(histogram_bins) % 2 == 0
    for index in range(0, len(histogram_bins), 2):
        assert histogram_bins[index].ndim == 1
        np.testing.assert_array_equal(histogram_bins[index], histogram_bins[index + 1])
    names = {path.name for path in destination.glob("*.png")}
    assert len(names) == 3 + 4 + 2 * 3
    assert {"success-curves.png", "loss-curves.png", "success-table.png",
            "official-range100-world-position-maps.png", "expanded3x200-relative-bowl-position-maps.png",
            "coverage-train-cream_cheese_1.png", "coverage-validation-cheese_minus_bowl.png"} <= names
