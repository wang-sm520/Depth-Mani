import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import depth_policy.collect_random as module
from depth_policy.episodes import EpisodeWriter, episode_index


def test_frozen_native_attempt_schedule_separates_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TARGETS", {"train": 2, "validation": 1})
    initial = tmp_path / "official.pt"
    initial.write_bytes(b"official")
    config = {"suite": "libero_goal", "task": module.TASK}
    official = np.arange(6, dtype=np.float64).reshape(2, 3)
    bank = module.create_bank(config, official, initial,
                              {"train": 3, "validation": 2},
                              {"train": 100, "validation": 200})
    assert [(entry["id"], entry["seed"], entry["split"]) for entry in bank["states"]] == [
        (0, 100, "train"), (1, 101, "train"), (2, 102, "train"),
        (3, 200, "validation"), (4, 201, "validation")]
    with pytest.raises(ValueError, match="overlap"):
        module.create_bank(config, official, initial,
                           {"train": 3, "validation": 2},
                           {"train": 100, "validation": 102})
    bank_path = tmp_path / "bank.json"
    module.atomic_json(bank_path, bank)
    assert module.load_bank(bank_path, config, official, initial) == bank
    config["task"] = "put_the_wine_bottle_on_top_of_the_cabinet"
    with pytest.raises(ValueError, match="cream cheese"):
        module.load_bank(bank_path, config, official, initial)


class FakeEnvironment:
    def __init__(self, fail_seed=None):
        self.fail_seed = fail_seed
        self.sim = SimpleNamespace(
            model=SimpleNamespace(get_xml=lambda: "<mujoco/>"),
            data=SimpleNamespace(get_joint_qpos=self.joint_qpos))
        self.env = SimpleNamespace(objects_dict={name: SimpleNamespace(joints=[name])
                                                 for name in (module.CHEESE, module.BOWL, "plate_1")})
        self.seed_value = None
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        if self.seed_value == self.fail_seed:
            raise RuntimeError("native reset failed")

    def get_sim_state(self):
        return np.asarray([float(self.seed_value), 0, 1], dtype=np.float64)

    def joint_qpos(self, name):
        x = {module.CHEESE: 0.01, module.BOWL: -0.02, "plate_1": 0.03}[name]
        return np.asarray([x, self.seed_value * 1e-6, 0.9, 1, 0, 0, 0])

    def check_success(self):
        return False

    def close(self):
        self.closed = True


def setup_fake_collector(tmp_path, monkeypatch, fail_seed=None):
    monkeypatch.setattr(module, "TARGETS", {"train": 2, "validation": 1})
    config = {"suite": "libero_goal", "task": module.TASK, "resolution": 4,
              "control_freq": 20, "settle_steps": 1, "teacher_replan_steps": 5,
              "openpi_root": str(tmp_path), "max_steps": 3}
    official = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    initial = tmp_path / "official.pt"
    initial.write_bytes(b"official")
    bank = module.create_bank(config, official, initial,
                              {"train": 3, "validation": 2},
                              {"train": 100, "validation": 200})
    bank_path = tmp_path / "attempts.json"
    module.atomic_json(bank_path, bank)
    monkeypatch.setattr(module, "task_and_states", lambda cfg: (None, official, initial))
    environments = []

    def make_env(cfg, seed):
        environment = FakeEnvironment(fail_seed)
        environments.append(environment)
        return environment, SimpleNamespace(language="Put the cream cheese on the bowl")

    monkeypatch.setattr(module, "make_env", make_env)
    monkeypatch.setattr(module, "Teacher", lambda cfg: SimpleNamespace(
        metadata={"policy_config": "pi05_libero", "checkpoint_path": "/test/pi05_libero"}))
    monkeypatch.setattr(module, "source_identity", lambda root: {})
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    seen = []

    def run_episode(environment, task, state, metadata, cfg, policy, output, stop):
        seen.append(metadata["seed"])
        writer = EpisodeWriter(output, {**metadata, "instruction": task.language}, state, "<mujoco/>")
        observation = {"state": np.zeros(8, dtype=np.float32),
                       "rgb_agentview": np.zeros((4, 4, 3), dtype=np.uint8),
                       "rgb_wrist": np.zeros((4, 4, 3), dtype=np.uint8),
                       "depth_agentview": np.ones((4, 4), dtype=np.float32),
                       "depth_wrist": np.ones((4, 4), dtype=np.float32)}
        for index in range(2):
            writer.append(observation, np.zeros(7), index, index / 20, state, index, np.zeros((2, 12)))
        success = metadata["seed"] != 100
        reason = "success" if success else "timeout"
        path = writer.finish(success, reason, state, [])
        return {"path": str(path), "success": success, "reason": reason, "steps": 2,
                "initial_state_id": metadata["initial_state_id"]}

    monkeypatch.setattr(module, "run_episode", run_episode)
    return config, bank_path, tmp_path / "teacher", seen, environments


def test_failures_consume_distinct_seeds_and_resumption_preserves_schema1(tmp_path, monkeypatch):
    config, bank_path, output, seen, environments = setup_fake_collector(tmp_path, monkeypatch)
    first = module.collect(config, bank_path, "train", 2, output, min_free_gib=0)
    assert first["successes"] == 1 and seen == [100, 101]
    second = module.collect(config, bank_path, "train", 2, output, min_free_gib=0)
    assert second["successes"] == 2 and seen == [100, 101, 102]
    third = module.collect(config, bank_path, "validation", 2, output, min_free_gib=0)
    assert third["successes"] == 1 and seen == [100, 101, 102, 200]
    assert all(environment.closed for environment in environments)
    ledger = json.loads((output / "attempt-index.json").read_text())
    rows = ledger["entries"]
    assert [row["status"] for row in rows] == ["timeout", "success", "success", "success"]
    assert len({row["initial_state_sha256"] for row in rows}) == 4
    assert rows[0]["relative_cream_cheese_to_bowl_xyz_m"] == pytest.approx([0.03, 0, 0])
    assert set(rows[0]["object_initial_poses"]) == {module.CHEESE, module.BOWL, "plate_1"}
    assert len([row for row in rows if row["split"] == "train" and row["status"] == "success"]) == 2
    assert len([row for row in rows if row["split"] == "validation" and row["status"] == "success"]) == 1
    records = episode_index(output)
    for record in records:
        with h5py.File(record["path"], "r") as stream:
            assert stream.attrs["schema_version"] == 1
            assert "rgb_agentview" in stream["steps"] and "depth_wrist" in stream["steps"]
            assert record["collection_bank_sha256"] == module.sha256_file(bank_path)
    assert module.collect(config, bank_path, "train", 1, output, min_free_gib=0)["attempts_this_run"] == 0


def test_pre_episode_reset_error_is_durable_and_blocks_silent_retry(tmp_path, monkeypatch):
    config, bank_path, output, seen, environments = setup_fake_collector(tmp_path, monkeypatch, fail_seed=100)
    with pytest.raises(RuntimeError, match="native reset failed"):
        module.collect(config, bank_path, "train", 1, output, min_free_gib=0)
    ledger = json.loads((output / "attempt-index.json").read_text())
    row = ledger["entries"][0]
    assert row["id"] == 0 and row["seed"] == 100 and row["status"] == "exception"
    assert "native reset failed" in row["error"]
    assert not seen and not episode_index(output) and environments[0].closed
    with pytest.raises(RuntimeError, match="require diagnosis"):
        module.collect(config, bank_path, "train", 1, output, min_free_gib=0)
    result = module.collect(config, bank_path, "train", 2, output, min_free_gib=0,
                            recovery_note="Seed 100 reset raised before simulation; preserve that failure")
    assert result["successes"] == 2 and seen == [101, 102]
    ledger = json.loads((output / "attempt-index.json").read_text())
    assert ledger["entries"][0]["status"] == "exception"
    assert ledger["recoveries"][0]["attempt_ids"] == [0]
    assert "Seed 100" in ledger["recoveries"][0]["note"]


def test_episode_validation_detects_scene_or_ledger_conflict(tmp_path, monkeypatch):
    config, bank_path, output, _, _ = setup_fake_collector(tmp_path, monkeypatch)
    module.collect(config, bank_path, "train", 1, output, min_free_gib=0)
    ledger_path = output / "attempt-index.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["entries"][0]["status"] = "exception"
    module.atomic_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="diagnosis"):
        module.collect(config, bank_path, "train", 1, output, min_free_gib=0)


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="Run with scripts/sim_cpu.sh")
def test_native_cream_cheese_reset_reproduces_state_xml_and_object_poses():
    config = module.load_config("configs/cream_cheese.json")
    environment, _ = module.make_env(config, 2026100000)
    try:
        observations = []
        for _ in range(2):
            np.random.seed(2026100000)
            environment.seed(2026100000)
            environment.reset()
            poses = module.object_poses(environment)
            observations.append((environment.get_sim_state().copy(), environment.sim.model.get_xml(), poses))
        np.testing.assert_array_equal(observations[0][0], observations[1][0])
        assert observations[0][1:] == observations[1][1:]
        assert set(observations[0][2]) == {module.CHEESE, module.BOWL, "wine_bottle_1", "plate_1"}
        assert len(module.relative_position(observations[0][2])) == 3
    finally:
        environment.close()
