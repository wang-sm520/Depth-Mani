import json
import os

import pytest

from depth_policy.common import atomic_json, sha256_file
from scripts import cream_cheese50k_experiment as coordinator


def test_frozen_banks_match_actual_attempt_schedule_and_data(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "ROOT", tmp_path)
    config = {"label": "cream", "attempt_bank": "attempts.json", "training_data": "teacher",
              "train_attempts": 2, "validation_attempts": 1,
              "groups": {"original": {"config": "original.json", "seed": 40, "count": 1},
                         "expanded": {"config": "expanded.json", "seed": 50, "count": 1}}}
    atomic_json(tmp_path / "attempts.json", {"schema_version": 1, "states": [
        {"id": 0, "split": "train", "seed": 10},
        {"id": 1, "split": "train", "seed": 11},
        {"id": 2, "split": "validation", "seed": 20}]})
    atomic_json(tmp_path / "teacher/attempt-index.json", {
        "bank_sha256": sha256_file(tmp_path / "attempts.json"),
        "entries": [{"id": 0, "initial_state_sha256": "teacher-state"}]})
    for group, settings in config["groups"].items():
        atomic_json(tmp_path / settings["config"], {"task": group})
        atomic_json(tmp_path / "data/cream_cheese/evaluation/cream" / f"{group}.json", {
            "mode": "random", "seed": settings["seed"], "config": {"task": group},
            "states": [{"id": 0, "seed": settings["seed"], "sha256": group,
                        "reset_and_settle_reproduced": True}]})
    frozen = coordinator.frozen_banks(config)
    assert set(frozen) == {"original", "expanded"}
    assert all(info["sha256"] == sha256_file(info["path"]) for info in frozen.values())
    path = tmp_path / "data/cream_cheese/evaluation/cream/expanded.json"
    bank = json.loads(path.read_text())
    bank["states"][0]["sha256"] = "teacher-state"
    atomic_json(path, bank)
    with pytest.raises(AssertionError):
        coordinator.frozen_banks(config)


def test_cream_protocol_counts_and_nonoverlapping_seeds():
    config = json.loads(coordinator.PROTOCOL.read_text())
    assert config["official_instruction"] == "Put the cream cheese on the bowl"
    assert sorted(group["count"] for group in config["groups"].values()) == [100, 200]
    assert config["checkpoint_steps"] == [10000, 20000, 30000, 40000, 50000]
    assert config["training_limit"] == 300 and config["validation_limit"] == 7
    seeds = set(range(config["train_seed"], config["train_seed"] + config["train_attempts"]))
    validation = set(range(config["validation_seed"], config["validation_seed"] + config["validation_attempts"]))
    assert not seeds & validation
    seeds |= validation
    for group in config["groups"].values():
        test = set(range(group["seed"], group["seed"] + group["count"]))
        assert not seeds & test
        seeds |= test


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="Run with scripts/sim_cpu.sh")
def test_policy_instruction_is_the_loaded_official_bddl_language():
    from depth_policy.common import load_config
    from depth_policy.simulation import make_env

    config = load_config("configs/cream_cheese_on.json")
    environment, task = make_env(config, 7)
    try:
        assert task.language == config["official_instruction"]
        assert environment.language_instruction.casefold() == task.language.casefold()
        assert task.language != "put the cream cheese in the bowl"
    finally:
        environment.close()
    config["official_instruction"] = "Put the cream cheese in the bowl"
    with pytest.raises(ValueError, match="differs from the loaded BDDL"):
        make_env(config, 7)
