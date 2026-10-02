import copy
import os

import numpy as np
import pytest

from depth_policy.evaluate_suite import check_records, object_quaternions, prepare_bank, state_hash


def test_object_quaternions_preserve_joint_orientation():
    from types import SimpleNamespace

    state = np.array([1, 2, 3, 0.5, 0.5, 0.5, 0.5])
    environment = SimpleNamespace(
        env=SimpleNamespace(objects_dict={"cream_cheese_1": SimpleNamespace(joints=["joint"])}),
        sim=SimpleNamespace(data=SimpleNamespace(get_joint_qpos=lambda joint: state)),
    )
    assert object_quaternions(environment) == {"cream_cheese_1": [0.5] * 4}


def test_state_hash_roundtrip():
    state = np.arange(17, dtype=np.float64) / 3
    assert state_hash(state) == state_hash(state.tolist())
    changed = state.copy()
    changed[3] += 1e-8
    assert state_hash(state) != state_hash(changed)


def test_official_bank_preserves_all_splits(tmp_path, monkeypatch):
    import depth_policy.evaluate_suite as module

    source = tmp_path / "source"
    source.write_text("source")
    states = np.arange(60, dtype=np.float64).reshape(3, 20)
    entries = [{"id": index, "sha256": state_hash(state), "split": split}
               for index, (state, split) in enumerate(zip(states, ["train", "validation", "test"], strict=True))]
    monkeypatch.setattr(module, "task_and_states", lambda config: (None, states, source))
    monkeypatch.setattr(module, "load_manifest", lambda *args: {"states": entries})
    bank = prepare_bank({"seed": 123, "manifest": str(source)}, "official", 3, 999)
    assert [entry["split"] for entry in bank["states"]] == ["train", "validation", "test"]
    assert [entry["seed"] for entry in bank["states"]] == [123, 124, 125]
    np.testing.assert_array_equal([entry["state"] for entry in bank["states"]], states)
    with pytest.raises(ValueError):
        prepare_bank({"seed": 123, "manifest": str(source)}, "official", 2, 999)


def test_evaluation_identity_and_no_silent_retries():
    entry = {"id": 4, "seed": 123, "sha256": "state", "split": "random-test"}
    policy = {"sha256": "checkpoint"}
    record = {"initial_state_id": 4, "seed": 123, "initial_state_sha256": "state",
              "split": "random-test", "teacher": policy, "evaluation_bank_sha256": "bank",
              "config": {"teacher_replan_steps": 4}, "termination_reason": "timeout"}
    assert check_records([record], {"states": [entry]}, policy, "bank", 4) == {4}
    for field, value in [("seed", 124), ("initial_state_sha256", "changed"),
                         ("teacher", {"sha256": "wrong"}), ("evaluation_bank_sha256", "wrong"),
                         ("split", "train"), ("termination_reason", "interrupted"),
                         ("record_images", False),
                         ("termination_reason", "exception")]:
        changed = copy.deepcopy(record)
        changed[field] = value
        with pytest.raises(ValueError):
            check_records([changed], {"states": [entry]}, policy, "bank", 4)
    with pytest.raises(ValueError):
        check_records([record, record], {"states": [entry]}, policy, "bank", 4)
    with pytest.raises(ValueError):
        check_records([record], {"states": [entry]}, policy, "bank", 5)


def test_custom_distribution_cannot_be_official():
    with pytest.raises(ValueError, match="cannot be labelled official"):
        prepare_bank({"evaluation_only": True}, "official", 50, 0)


def test_evaluation_configuration_cannot_collect_training(monkeypatch):
    from depth_policy.collect import main

    monkeypatch.setattr("sys.argv", ["collect", "--config", "configs/wine_10cm_eval.json"])
    with pytest.raises(ValueError, match="cannot collect training"):
        main()


def test_10cm_bddl_changes_only_wine_region():
    import json
    from pathlib import Path
    from depth_policy.common import sha256_file

    config = json.loads(Path("configs/wine_10cm_eval.json").read_text())
    changed = Path(config["evaluation_bddl"])
    original = Path(config["libero_root"]) / "libero/libero/bddl_files/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.bddl"
    if not original.exists():
        pytest.skip("Local LIBERO source unavailable")
    assert sha256_file(changed) == config["evaluation_bddl_sha256"]
    assert changed.read_text().replace("(-0.275 -0.125 -0.125 0.025)",
                                      "(-0.21000000000000002 -0.060000000000000005 -0.19 -0.04)") == original.read_text()
    radius = 0.025
    np.testing.assert_allclose([-0.275 + radius, -0.125 + radius],
                               config["evaluation_center_bounds"]["wine_bottle_1"]["minimum_xy_m"])
    np.testing.assert_allclose([-0.125 - radius, 0.025 - radius],
                               config["evaluation_center_bounds"]["wine_bottle_1"]["maximum_xy_m"])


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="Run with scripts/sim_cpu.sh")
def test_10cm_reset_matches_deployment_restoration():
    from depth_policy.common import load_config

    bank = prepare_bank(load_config("configs/wine_10cm_eval.json"), "random", 1, 2026092384)
    assert len(bank["states"]) == 1
    assert bank["states"][0]["reset_and_settle_reproduced"]
