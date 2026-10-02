from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from airbot_depth import audit
from airbot_depth.convert import episode_split


def contract_fixture(count=2, fraction=0.1):
    indices = list(range(count))
    splits = episode_split(indices, fraction, 20260924)
    source_code = {"depth.py": "depth-sha", "convert.py": "convert-sha"}
    manifest = {"schema_version": 1, "status": "complete", "source": {
        "selected_episode_indices": indices, "total_episodes": count}, "robot_type": "airbot_play_follower",
        "fps": 25.0, "camera_keys": ["head", "wrist"], "state_dim": 7, "action_dim": 7,
        "state_names": [f"joint{index}" for index in range(6)] + ["gripper"],
        "action_names": [f"joint{index}" for index in range(6)] + ["gripper"],
        "state_units": ["rad"] * 6 + ["m"], "action_units": ["rad"] * 6 + ["m"],
        "action_semantics": "absolute_joint_position", "prompt": "pick up the bag",
        "depth_config": {"image_size": 16}, "depth_provenance": {"saved": True},
        "model_input_quantization": "float32_to_float16_to_float32", "split_seed": 20260924,
        "validation_fraction": fraction, "episode_splits": {str(key): value for key, value in splits.items()},
        "source_code": source_code,
        "episodes": [{"episode_index": index, "split": splits[index], "path": f"episode_{index}.h5",
                      "frames": 3, "sha256": f"episode-sha-{index}"} for index in indices]}
    identity = {key: manifest[key] for key in audit.IDENTITY_FIELDS}
    manifest["conversion_signature"] = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    train = sum(value == "train" for value in splits.values())
    run = {"status": "complete", "manifest_sha256": "manifest-sha", "manifest": deepcopy(manifest),
           "initialization": "all_policy_parameters_random", "config": {"steps": 10}, "completed_steps": 10,
           "train_episodes": train, "validation_episodes": count - train, "train_frames": train * 3,
           "validation_frames": (count - train) * 3, "source": {"airbot_depth/model.py": "model-sha"}}
    return manifest, run


def test_formal_split_and_smoke_scope_are_distinct():
    manifest, run = contract_fixture(200)
    report = audit.validate_contract(manifest, run, "manifest-sha")
    assert report["formal_180_train_20_validation"]
    assert (report["train_episodes"], report["validation_episodes"]) == (180, 20)
    manifest, run = contract_fixture(2)
    report = audit.validate_contract(manifest, run, "manifest-sha")
    assert report["whole_episode_split_verified"]
    assert not report["formal_200_episode_dataset"]
    manifest, run = contract_fixture(200, fraction=0.2)
    with pytest.raises(ValueError, match="180/20"):
        audit.validate_contract(manifest, run, "manifest-sha")


@pytest.mark.parametrize("mutation", ["hash", "signature", "split", "leak"])
def test_contract_rejects_changed_dataset_or_episode_leakage(mutation):
    manifest, run = contract_fixture()
    if mutation == "hash":
        run["manifest_sha256"] = "different"
    elif mutation == "signature":
        manifest["conversion_signature"] = "different"
    elif mutation == "split":
        manifest["episodes"][0]["split"] = "validation" if manifest["episodes"][0]["split"] == "train" else "train"
    else:
        manifest["episodes"][1]["sha256"] = manifest["episodes"][0]["sha256"]
    run["manifest"] = deepcopy(manifest)
    with pytest.raises(ValueError):
        audit.validate_contract(manifest, run, "manifest-sha")


def test_report_uses_native_units_and_hold_state_baseline():
    target = np.zeros((2, 7), dtype=np.float32)
    prediction = np.asarray([[1, 2, 3, 4, 5, 6, 0.001], [3, 4, 5, 6, 7, 8, 0.003]], dtype=np.float32)
    report = audit.action_error_summary(prediction, target)
    np.testing.assert_allclose(report["joint_mae_rad"], [2, 3, 4, 5, 6, 7])
    assert report["joint_mean_mae_rad"] == pytest.approx(4.5)
    assert report["gripper_mae_mm"] == pytest.approx(2.0)
    assert report["validation_frames"] == 2
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.action_error_summary(prediction, target)


def test_first_action_metric_ignores_future_chunk_and_uses_every_frame():
    class FixedPolicy(nn.Module):
        def forward(self, state, tokens, images):
            values = torch.full((len(state), 3, 7), 10000.0)
            values[:, 0] = 0.5
            return values

    policy = SimpleNamespace(model=FixedPolicy(), device=torch.device("cpu"), statistics={
        "action_mean": np.zeros(7, dtype=np.float32), "action_std": np.ones(7, dtype=np.float32) * 0.02})
    dataset = [{"state": torch.zeros(7), "tokens": torch.zeros(2, dtype=torch.long),
                "images": torch.zeros(2, 2, 16, 16), "mask": torch.tensor([True, False, False]),
                "action": torch.zeros(3, 7)} for _ in range(3)]
    result = audit.evaluate_first_actions(policy, dataset, np.zeros((3, 7)), batch_size=2)
    assert result["validation_frames"] == 3
    assert result["joint_mean_mae_rad"] == pytest.approx(0.01)
    assert result["gripper_mae_mm"] == pytest.approx(10.0)


def test_source_code_change_is_reported_and_model_change_rejected(tmp_path, monkeypatch):
    manifest, run = contract_fixture()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    directory = tmp_path / "airbot_depth"
    directory.mkdir()
    for name in ("convert.py", "depth.py", "model.py"):
        (directory / name).write_text(name)
    digests = {"convert.py": audit.CONVERTER_LOCK_CHANGE[1], "depth.py": "depth-sha", "model.py": "model-sha"}
    monkeypatch.setattr(audit, "sha256_file", lambda path: digests.get(Path(path).name, "audit-sha"))
    manifest["source_code"]["convert.py"] = audit.CONVERTER_LOCK_CHANGE[0]
    report = audit.audit_source_code(manifest, run)
    assert report["changed_files"] == ["airbot_depth/convert.py"]
    assert not report["all_saved_files_unchanged"]
    assert any("lock" in item.get("note", "") for item in report["files"])
    digests["model.py"] = "changed-model-sha"
    with pytest.raises(ValueError, match="Inference implementation changed"):
        audit.audit_source_code(manifest, run)


def test_roundtrip_checks_depth_and_full_action_chunk(tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.zeros((2, 2, 16, 16), dtype=np.float32)
    depth[:, 0] = 0.25
    depth[:, 1] = 1
    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as stream:
        stream["timestamp"] = [0.0, 0.04, 0.08]
        stream["state"] = np.zeros((3, 7), dtype=np.float32)
        stream["depth"] = np.broadcast_to(depth, (3, 2, 2, 16, 16))
    manifest = {"depth_config": {"image_size": 16}, "depth_provenance": {"saved": True},
                "camera_keys": ["head", "wrist"],
                "episodes": [{"episode_index": 7, "path": path.name, "frames": 3, "split": "validation"}]}

    class Transform:
        config = manifest["depth_config"]
        def provenance(self):
            return manifest["depth_provenance"]
        def __call__(self, images):
            return depth.copy()

    class Policy:
        checkpoint_path = Path("best.pt")
        checkpoint_sha256 = "policy-sha"
        camera_keys = ["head", "wrist"]
        _depth_transform = Transform()
        delta = 0.0
        def infer_depth(self, state, input_depth):
            return np.zeros((4, 7), dtype=np.float32)
        def infer_rgb(self, state, input_rgb):
            result = np.zeros((4, 7), dtype=np.float32)
            result[-1, -1] = self.delta
            return result

    source = SimpleNamespace(camera_keys=["head", "wrist"],
                             rgb_frames=lambda index, key, stamps: iter((rgb.copy(), stamp) for stamp in stamps))
    policy = Policy()
    result, examples = audit.audit_roundtrip(policy, source, tmp_path, manifest, [7], 1e-6)
    assert result["passed"] and result["depth_max_abs_difference"] == 0
    assert len(examples) == 3
    assert result["camera_field_order_verified"]
    assert set(result["frames"][0]["camera_rgb_inputs"]) == {"head", "wrist"}
    policy.delta = 0.001
    with pytest.raises(ValueError, match="Roundtrip mismatch"):
        audit.audit_roundtrip(policy, source, tmp_path, manifest, [7], 1e-6)


def test_joint_limit_discrepancy_is_reported_without_clipping():
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 1] = [-0.5, 0.18, 0.25]
    original = actions.copy()
    result = audit.joint_limit_summary(actions)
    assert result["above_reference_count"] == 2
    assert result["raw_max_rad"] == 0.25
    np.testing.assert_array_equal(actions, original)
    episodes = audit.episode_joint_limit_summary({0: actions[:1], 9: actions[1:]})
    assert episodes["episode_count_above_reference"] == 1
    assert episodes["episode_indices_above_reference"] == [9]


def test_existing_report_path_is_refused_before_reading_data(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError, match="existing report"):
        audit.main(["--data", "/missing/data", "--run", "/missing/run", "--output", str(output)])
