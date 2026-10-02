from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from scripts.audit_airbot_resume import (
    RESTORED_FIELDS, checkpoint_origin, original, verify_metrics, verify_optimizer_step,
    verify_restoration, verify_run_chain,
)
from scripts.resume_airbot_depth import NUMERICAL_FLAGS, state_digest, training_state_digests


def chain_records():
    indices = list(range(200))
    split = original.episode_split(indices, 0.1, 20260924)
    manifest = {key: None for key in original.IDENTITY_FIELDS}
    manifest.update(schema_version=1, status="complete", source={"selected_episode_indices": indices, "total_episodes": 200},
                    state_dim=7, action_dim=7, state_units=["rad"] * 6 + ["m"], action_units=["rad"] * 6 + ["m"],
                    action_semantics="absolute_joint_position", validation_fraction=0.1, split_seed=20260924,
                    episode_splits={str(key): value for key, value in split.items()}, fps=25.0)
    manifest["episodes"] = [{"episode_index": index, "path": f"ep{index}.h5", "sha256": f"{index:064x}",
                            "frames": 1, "split": split[index]} for index in indices]
    identity = {key: manifest[key] for key in original.IDENTITY_FIELDS}
    manifest["conversion_signature"] = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    parent = {"status": "complete", "initialization": "all_policy_parameters_random", "manifest": manifest,
              "manifest_sha256": "manifest", "completed_steps": 50000,
              "config": {"steps": 50000, "output": "parent", "batch_size": 4, "eval_every": 1000, "checkpoint_every": 10000},
              "train_episodes": 180, "validation_episodes": 20, "train_frames": 180, "validation_frames": 20}
    identities = {"parent_run_sha256": "a" * 64, "parent_audit_sha256": "b" * 64,
                  "parent_checkpoint_sha256": "c" * 64, "parent_best_checkpoint_sha256": "d" * 64}
    audit = {"status": "passed", "formal_50000_step_training": True, "training_complete_steps": 50000,
             "run_sha256": identities["parent_run_sha256"], "manifest_sha256": "manifest",
             "latest_checkpoint_sha256": identities["parent_checkpoint_sha256"],
             "source_code": {"all_saved_files_unchanged": True},
             "roundtrip": {"passed": True, "runtime_depth_provenance_exact_match": True},
             "validation": {"checkpoints": [{"name": "best.pt", "step": 43000,
                                             "checkpoint_sha256": identities["parent_best_checkpoint_sha256"],
                                             "validation_loss": 0.01}]}}
    run = deepcopy(parent)
    run.update(initialization="resume_full_training_state", root_initialization="all_policy_parameters_random",
               start_step=50000, completed_steps=100000)
    run["config"].update(steps=100000, output="continued")
    run["resume"] = {"schema_version": 1, "start_step": 50000, "target_step": 100000,
                     "parent_best_step": 43000, "parent_best_validation_loss": 0.01, **identities,
                     "restored_fields": [*RESTORED_FIELDS, "best_validation_loss", "numerical_flags"]}
    return run, parent, audit, manifest, identities


def training_payload(step=50000):
    return {
        "step": step, "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}},
                      "param_groups": [{"params": [0]}]},
        "sampler_rng": torch.Generator().manual_seed(17).get_state(), "torch_rng": torch.get_rng_state(),
        "python_rng": random.Random(3).getstate(), "numpy_rng": np.random.RandomState(4).get_state(), "cuda_rng": None,
        "environment": {"device": "cpu", **{name: False for name in NUMERICAL_FLAGS}},
    }


def restoration_records():
    payload = training_payload()
    digests = training_state_digests(payload)
    generator = torch.Generator()
    generator.set_state(payload["sampler_rng"])
    before = state_digest(generator.get_state())
    expected = torch.randint(180, (4,), generator=generator).tolist()
    after = state_digest(generator.get_state())
    flags = {name: payload["environment"][name] for name in NUMERICAL_FLAGS}
    summary = {"count": 1, "min": 50000.0, "max": 50000.0}
    run = {"train_frames": 180, "config": {"batch_size": 4},
           "resume_verification": {"schema_version": 1, "all_restored_exact": True,
                                   "saved_state_digests": digests, "restored_state_digests": deepcopy(digests),
                                   "optimizer_checkpoint": summary, "optimizer_after_restore": summary,
                                   "optimizer_after_first_step": {"count": 1, "min": 50001.0, "max": 50001.0},
                                   "runtime_flags": {"saved": flags, "restored": flags, "exact_match": True},
                                   "first_batch": {"global_step": 50001, "expected_indices": expected, "actual_indices": expected,
                                                   "exact_match": True, "sampler_rng_before_sha256": before,
                                                   "sampler_rng_after_sha256": after, "expected_sampler_rng_after_sha256": after}}}
    return run, payload


def test_parent_contract_is_validated_without_rewriting_continuation_initialization():
    run, parent, audit, manifest, identities = chain_records()
    contract = verify_run_chain(run, parent, audit, manifest, "manifest", identities)
    assert contract["formal_180_train_20_validation"] is True
    assert run["initialization"] == "resume_full_training_state"
    assert parent["initialization"] == "all_policy_parameters_random"


@pytest.mark.parametrize("change", ["parent_hash", "parent_audit", "root_initialization", "steps", "formal_flag", "hyperparameter"])
def test_continuation_rejects_forged_or_changed_parent_contract(change):
    run, parent, audit, manifest, identities = chain_records()
    if change == "parent_hash":
        run["resume"]["parent_checkpoint_sha256"] = "different-parent"
    elif change == "parent_audit":
        audit["latest_checkpoint_sha256"] = "different-parent"
    elif change == "root_initialization":
        run["root_initialization"] = "pretrained"
    elif change == "steps":
        run["completed_steps"] = 99999
    elif change == "formal_flag":
        audit["formal_50000_step_training"] = False
    else:
        run["config"]["batch_size"] = 8
    with pytest.raises(ValueError):
        verify_run_chain(run, parent, audit, manifest, "manifest", identities)


def test_restoration_recomputes_parent_digests_and_first_sampler_batch():
    run, payload = restoration_records()
    assert verify_restoration(run, payload)["first_batch_recomputed"] is True
    run["resume_verification"]["first_batch"]["actual_indices"] = [0] * 4
    with pytest.raises(ValueError, match="minibatch"):
        verify_restoration(run, payload)


@pytest.mark.parametrize("change", ["missing_optimizer", "reset_step", "wrong_digest", "missing_rng", "wrong_flags"])
def test_restoration_does_not_accept_incomplete_or_reset_training_state(change):
    run, payload = restoration_records()
    if change == "missing_optimizer":
        del payload["optimizer"]
    elif change == "reset_step":
        payload["optimizer"]["state"][0]["step"] = torch.tensor(0.0)
    elif change == "wrong_digest":
        run["resume_verification"]["restored_state_digests"]["optimizer"] = "fabricated"
    elif change == "missing_rng":
        del payload["numpy_rng"]
    else:
        run["resume_verification"]["runtime_flags"] = {"saved": {}, "restored": {}, "exact_match": True}
    with pytest.raises(ValueError):
        verify_restoration(run, payload)


def test_final_optimizer_requires_all_parameters_at_exact_global_step():
    payload = training_payload(100000)
    assert verify_optimizer_step(payload, 100000)["min"] == 100000
    payload["optimizer"]["state"][0]["step"] = torch.tensor(50000.0)
    with pytest.raises(ValueError, match="Optimizer step"):
        verify_optimizer_step(payload, 100000)


def test_parent_best_keeps_original_config_and_requires_exact_inherited_sha():
    payload = training_payload(43000)
    manifest = {"identity": "original-data"}
    parent = {key: {} for key in ("model_config", "statistics", "vocabulary", "depth_config", "depth_provenance", "source")}
    parent.update(config={"steps": 50000}, environment=payload["environment"], completed_steps=50000,
                  action_semantics="absolute_joint_position", image_input_precision="float16_then_float32")
    payload.update({key: parent[key] for key in parent if key != "completed_steps"},
                   manifest=manifest, manifest_sha256="manifest", validation_loss=0.1, best_validation_loss=0.1)
    run = {"resume": {"parent_run_sha256": "parent-run"}}
    result = checkpoint_origin(payload, "best-sha", run, parent, manifest, "manifest", {"best-sha": 43000})
    assert result == {"phase": "parent", "origin_run_target_steps": 50000, "origin_run_sha256": "parent-run"}
    assert payload["config"]["steps"] == 50000
    with pytest.raises(ValueError, match="exact inherited"):
        checkpoint_origin(payload, "other-sha", run, parent, manifest, "manifest", {"best-sha": 43000})


def test_metrics_require_original_byte_prefix_and_complete_continuation_cadence(tmp_path):
    parent, current = tmp_path / "parent", tmp_path / "current"
    parent.mkdir()
    current.mkdir()
    prefix = b'{"step":50000,"train_loss":1.0,"validation_loss":1.0}\n'
    (parent / "metrics.jsonl").write_bytes(prefix)
    suffix = [{"step": step, "train_loss": 0.5, **({"validation_loss": 0.5} if step == 50100 else {})}
              for step in [50001, 50025, 50050, 50075, 50100]]
    encoded = b"".join((json.dumps(item) + "\n").encode() for item in suffix)
    (current / "metrics.jsonl").write_bytes(prefix + encoded)
    run = {"start_step": 50000, "completed_steps": 50100, "config": {"eval_every": 100, "checkpoint_every": 10000},
           "resume": {"parent_metrics_bytes": len(prefix), "parent_metrics_sha256": hashlib.sha256(prefix).hexdigest(),
                      "parent_metrics_records": 1}}
    assert verify_metrics(current, parent, run)[1]["exact_parent_prefix"] is True
    (current / "metrics.jsonl").write_bytes(prefix + b"".join((json.dumps(item) + "\n").encode() for item in suffix[1:]))
    with pytest.raises(ValueError, match="logged optimizer steps"):
        verify_metrics(current, parent, run)
    (current / "metrics.jsonl").write_bytes(prefix.replace(b"1.0", b"1.1") + encoded)
    with pytest.raises(ValueError, match="exact parent byte prefix"):
        verify_metrics(current, parent, run)
