from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from airbot_depth.common import atomic_json, sha256_file
from airbot_depth.depth import canonical_depth_config
from airbot_depth.model import AirbotPolicyModel
from airbot_depth.train import main as train_main
from scripts import resume_airbot_depth as resume


def make_dataset(root):
    root.mkdir()
    prompt = "pick up the red paper bag"
    configuration = canonical_depth_config({"image_size": 32})
    manifest = {"schema_version": 1, "status": "complete", "fps": 25, "robot_type": "airbot_play_follower",
                "camera_keys": ["observation.images.head", "observation.images.wrist"],
                "state_dim": 7, "action_dim": 7,
                "state_names": [f"joint{index}.pos" for index in range(1, 7)] + ["eef.pos"],
                "action_names": [f"joint{index}.pos" for index in range(1, 7)] + ["eef.pos"],
                "action_semantics": "absolute_joint_position", "prompt": prompt,
                "depth_config": configuration, "depth_provenance": {"test_fixture": True, "config": configuration},
                "source": {"selected_episode_indices": [0, 1, 2]}, "episodes": []}
    for index, (length, split) in enumerate([(4, "train"), (3, "train"), (3, "validation")]):
        path = root / f"episode_{index}.h5"
        state = np.linspace(-0.2, 0.5, length * 7, dtype=np.float32).reshape(length, 7) + index * 0.01
        action = state + np.linspace(0.002, 0.02, length, dtype=np.float32)[:, None]
        depth = np.zeros((length, 2, 2, 32, 32), dtype=np.float32)
        depth[:, :, 0] = np.linspace(0, 1, 32 * 32).reshape(32, 32)
        depth[:, :, 1] = 1
        with h5py.File(path, "w") as stream:
            stream.attrs["episode_index"] = index
            stream.attrs["instruction"] = prompt
            stream["state"], stream["action"], stream["depth"] = state, action, depth
        manifest["episodes"].append({"episode_index": index, "frames": length, "split": split,
                                     "path": path.name, "sha256": sha256_file(path), "instruction": prompt})
    atomic_json(root / "manifest.json", manifest)
    return manifest


def train_fixture(data, output, steps):
    train_main(["--data", str(data), "--output", str(output), "--steps", str(steps),
                "--device", "cpu", "--batch-size", "2", "--horizon", "3", "--seed", "12",
                "--checkpoint-every", "2", "--eval-every", "2", "--threads", "2"])


def make_parent_audit(directory, report_path):
    run = json.loads((directory / "run.json").read_text())
    rows = []
    for path in [*sorted(directory.glob("step_*.pt")), directory / "best.pt"]:
        payload = torch.load(path, weights_only=False, map_location="cpu")
        rows.append({"name": path.name, "step": payload["step"], "checkpoint_sha256": sha256_file(path),
                     "validation_loss": payload["validation_loss"]})
    # Synthetic source data has no RGB model download; this fixture represents
    # the audit boundary already independently tested by test_airbot_audit.py.
    audit = {"schema_version": 1, "status": "passed", "run": str(directory),
             "run_sha256": sha256_file(directory / "run.json"), "manifest_sha256": run["manifest_sha256"],
             "latest_checkpoint_sha256": sha256_file(directory / "latest.pt"),
             "training_complete_steps": run["completed_steps"],
             "source_code": {"all_saved_files_unchanged": True},
             "roundtrip": {"passed": True, "runtime_depth_provenance_exact_match": True},
             "validation": {"checkpoints": rows}}
    atomic_json(report_path, audit)
    return audit


@pytest.fixture(scope="module")
def parent_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume-parent")
    data, parent = root / "data", root / "parent"
    make_dataset(data)
    train_fixture(data, parent, 4)
    audit_path = root / "parent-audit.json"
    audit = make_parent_audit(parent, audit_path)
    return {"root": root, "data": data, "parent": parent, "audit_path": audit_path, "audit": audit,
            "run": json.loads((parent / "run.json").read_text()),
            "latest": torch.load(parent / "latest.pt", map_location="cpu", weights_only=False),
            "best": torch.load(parent / "best.pt", map_location="cpu", weights_only=False)}


def test_continuous_and_resumed_training_states_are_exactly_equal(parent_fixture):
    fixture = parent_fixture
    full, continued = fixture["root"] / "continuous", fixture["root"] / "continued"
    train_fixture(fixture["data"], full, 6)
    resume.main(["--parent-run", str(fixture["parent"]), "--parent-audit", str(fixture["audit_path"]),
                 "--output", str(continued), "--steps", "6", "--device", "cpu", "--threads", "2"])
    expected = torch.load(full / "latest.pt", map_location="cpu", weights_only=False)
    actual = torch.load(continued / "latest.pt", map_location="cpu", weights_only=False)
    # Includes every model parameter, every AdamW moment and step, the sampler,
    # Python, NumPy, Torch CPU RNG, and the saved CUDA RNG field (None on CPU).
    assert resume.training_state_digests(actual) == resume.training_state_digests(expected)
    assert actual["step"] == 6 and actual["start_step"] == 4
    assert actual["initialization"] == "resume_full_training_state"
    assert actual["root_initialization"] == "all_policy_parameters_random"
    verification = actual["resume_verification"]
    assert verification["saved_state_digests"] == resume.training_state_digests(fixture["latest"])
    assert verification["restored_state_digests"] == verification["saved_state_digests"]
    assert verification["optimizer_before_restore"] == {"count": 0, "min": None, "max": None}
    assert verification["optimizer_after_restore"]["min"] == verification["optimizer_after_restore"]["max"] == 4
    assert verification["optimizer_after_first_step"]["min"] == verification["optimizer_after_first_step"]["max"] == 5
    sampler = torch.Generator().manual_seed(12)
    for _ in range(4):
        torch.randint(7, (2,), generator=sampler)
    next_batch = torch.randint(7, (2,), generator=sampler).tolist()
    assert verification["first_batch"]["expected_indices"] == next_batch
    assert verification["first_batch"]["actual_indices"] == next_batch
    assert verification["first_batch"]["sampler_rng_after_sha256"] == resume.state_digest(sampler.get_state())
    for name in ("step_000002.pt", "step_000004.pt"):
        assert sha256_file(continued / name) == sha256_file(fixture["parent"] / name)
    assert sha256_file(continued / "parent_best.pt") == sha256_file(fixture["parent"] / "best.pt")
    original_prefix = (fixture["parent"] / "metrics.jsonl").read_bytes()
    assert (continued / "metrics.jsonl").read_bytes().startswith(original_prefix)
    suffix = (continued / "metrics.jsonl").read_bytes()[len(original_prefix):]
    records = [json.loads(line) for line in suffix.decode().splitlines()]
    assert records[0]["step"] == 5 and records[-1]["step"] == 6
    for record in records:
        assert record["elapsed_s"] == pytest.approx(fixture["run"]["elapsed_s"] + record["segment_elapsed_s"])
    best = torch.load(continued / "best.pt", map_location="cpu", weights_only=False)
    segment_best = torch.load(continued / "best_continuation.pt", map_location="cpu", weights_only=False)
    assert best["validation_loss"] == min(fixture["run"]["best_validation_loss"], segment_best["validation_loss"])
    assert segment_best["step"] == 6
    run = json.loads((continued / "run.json").read_text())
    assert run["status"] == "complete" and run["completed_steps"] == 6
    assert run["source"] == fixture["run"]["source"]
    assert run["driver_source"] == {"scripts/resume_airbot_depth.py": sha256_file(resume.__file__)}


def metadata_arguments(fixture):
    return {"run_sha": sha256_file(fixture["parent"] / "run.json"),
            "latest_sha": sha256_file(fixture["parent"] / "latest.pt"),
            "best_sha": sha256_file(fixture["parent"] / "best.pt"),
            "manifest": deepcopy(fixture["run"]["manifest"]), "manifest_sha": fixture["run"]["manifest_sha256"],
            "current_source": deepcopy(fixture["run"]["source"]), "parent_run": fixture["parent"]}


@pytest.mark.parametrize("mutation", ["data", "hyperparameters", "source", "audit", "checkpoint_hash"])
def test_changed_parent_identity_is_rejected(parent_fixture, mutation):
    fixture = parent_fixture
    parent, audit = deepcopy(fixture["run"]), deepcopy(fixture["audit"])
    latest, best = fixture["latest"], fixture["best"]
    kwargs = metadata_arguments(fixture)
    if mutation == "data":
        kwargs["manifest"]["prompt"] = "a different task"
    elif mutation == "hyperparameters":
        parent["config"]["learning_rate"] *= 2
    elif mutation == "source":
        kwargs["current_source"]["airbot_depth/model.py"] = "changed"
    elif mutation == "audit":
        audit["status"] = "failed"
    else:
        kwargs["latest_sha"] = "changed"
    with pytest.raises(ValueError):
        resume.validate_parent_metadata(parent, audit, latest, best, **kwargs)


@pytest.mark.parametrize("missing", ["optimizer", "sampler_rng", "torch_rng", "python_rng", "numpy_rng", "cuda_rng"])
def test_missing_full_training_state_is_rejected(parent_fixture, missing):
    payload = parent_fixture["latest"].copy()
    del payload[missing]
    with pytest.raises(ValueError, match="lacks full training state"):
        resume.training_state_digests(payload)


def test_changed_optimizer_hyperparameters_or_steps_are_rejected(parent_fixture):
    fixture = parent_fixture
    model = AirbotPolicyModel(**fixture["run"]["model_config"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.0001)
    with pytest.raises(ValueError, match="hyperparameters"):
        resume.restore_training_state(fixture["latest"], model, optimizer, torch.Generator(), "cpu")
    payload = fixture["latest"].copy()
    payload["optimizer"] = deepcopy(payload["optimizer"])
    next(iter(payload["optimizer"]["state"].values()))["step"] = torch.tensor(0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    with pytest.raises(ValueError, match="step counters"):
        resume.restore_training_state(payload, model, optimizer, torch.Generator(), "cpu")


def test_numerical_runtime_drift_is_rejected(parent_fixture):
    environment = parent_fixture["run"]["environment"]
    with pytest.raises(ValueError, match="thread count"):
        resume.restore_numerical_environment(environment, "cpu", 3, 2)
    altered = deepcopy(environment)
    altered["torch"] = "a different numerical runtime"
    with pytest.raises(ValueError, match="runtime differs"):
        resume.restore_numerical_environment(altered, "cpu", 2, 2)


def test_existing_output_is_rejected_before_parent_reads(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="new directory"):
        resume.main(["--parent-run", "/missing/parent", "--parent-audit", "/missing/audit",
                     "--output", str(output)])
