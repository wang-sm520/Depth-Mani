from copy import deepcopy
import hashlib

import pytest

from scripts.export_airbot_depth import (
    CORE_FILES, INFERENCE_KEYS, RUNTIME_FILES, export_bundle, inference_payload,
    verify_audit_claims, verify_runtime_sources,
)


def records(formal=False):
    train = list(range(180)) if formal else [0]
    validation = list(range(180, 200)) if formal else [199]
    steps = 50000 if formal else 10
    payload = {
        "config": {"steps": steps}, "step": 10, "manifest_sha256": "manifest",
        "manifest": {"source": {"selected_episode_indices": train + validation, "total_episodes": 200}},
    }
    audit = {
        "schema_version": 1, "status": "passed", "manifest_sha256": "manifest",
        "validation": {"checkpoints": [{"checkpoint_sha256": "selected", "step": 10}]},
        "roundtrip": {"passed": True, "runtime_depth_provenance_exact_match": True,
                      "checkpoint_sha256": "selected"},
        "training_complete_steps": steps, "formal_50000_step_training": formal,
        "dataset": {"train_episode_indices": train, "validation_episode_indices": validation,
                    "train_episodes": len(train), "validation_episodes": len(validation),
                    "formal_200_episode_dataset": formal, "formal_180_train_20_validation": formal},
        "source_code": {"formal_strict_source_equality": formal, "all_saved_files_unchanged": formal},
    }
    return payload, audit


def test_smoke_export_cannot_be_mislabelled_formal_and_best_step_remains_explicit():
    payload, audit = records()
    scope = verify_audit_claims(payload, audit, "selected")
    assert scope["formal_training_run"] is False
    assert scope["run_completed_steps"] == 10
    payload, audit = records(formal=True)
    scope = verify_audit_claims(payload, audit, "selected")
    assert scope["formal_training_run"] is True
    assert scope["checkpoint_training_step"] == 10
    assert scope["run_completed_steps"] == 50000
    audit["source_code"]["all_saved_files_unchanged"] = False
    with pytest.raises(ValueError, match="strict unchanged-source"):
        verify_audit_claims(payload, audit, "selected")


@pytest.mark.parametrize("change", ["checkpoint", "status", "step", "formal_label", "split", "roundtrip"])
def test_export_rejects_unverified_or_inconsistent_evidence(change):
    payload, audit = records()
    digest = "selected"
    if change == "checkpoint":
        digest = "another-checkpoint"
    elif change == "status":
        audit["status"] = "failed"
    elif change == "step":
        payload["step"] = 5
    elif change == "formal_label":
        audit["formal_50000_step_training"] = True
    elif change == "split":
        audit["dataset"]["validation_episode_indices"] = [0]
    else:
        audit["roundtrip"]["runtime_depth_provenance_exact_match"] = False
    with pytest.raises(ValueError):
        verify_audit_claims(payload, audit, digest)


def test_inference_checkpoint_omits_optimizer_and_rng_without_losing_policy_inputs():
    payload = {key: {"saved": key} for key in INFERENCE_KEYS}
    payload.update(optimizer={"large_training_state": True}, sampler_rng=b"rng", cuda_rng=[b"gpu_rng"])
    result = inference_payload(payload, "original-sha", {"formal_training_run": False})
    assert set(result) == set(INFERENCE_KEYS) | {"inference_export"}
    assert result["model"] == payload["model"]
    assert result["inference_export"]["source_checkpoint_sha256"] == "original-sha"
    assert result["inference_export"]["training_state_removed"] == ["cuda_rng", "optimizer", "sampler_rng"]


def test_runtime_sources_must_match_audit_and_frozen_core_training_bytes(tmp_path):
    payload = {"source": {}, "depth_provenance": {}}
    audit = {"source_code": {"files": [], "depth_and_policy_model_unchanged": True}}
    for relative in RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {relative}\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payload["source"][relative] = digest
        audit["source_code"]["files"].append({"path": relative, "saved_sha256": digest, "current_sha256": digest})
    payload["depth_provenance"]["implementation_sha256"] = payload["source"]["airbot_depth/depth.py"]
    assert len(verify_runtime_sources(payload, audit, tmp_path)) == len(RUNTIME_FILES)
    # An audited policy wrapper update is acceptable for a smoke export.
    payload["source"]["airbot_depth/policy.py"] = "previous-policy-hash"
    verify_runtime_sources(payload, audit, tmp_path)
    for relative in CORE_FILES:
        changed = deepcopy(payload)
        changed["source"][relative] = "different-training-source"
        with pytest.raises(ValueError, match="Core inference source"):
            verify_runtime_sources(changed, audit, tmp_path)
    (tmp_path / "airbot_depth/policy.py").write_text("# modified after audit\n")
    with pytest.raises(ValueError, match="changed after the audit"):
        verify_runtime_sources(payload, audit, tmp_path)


def test_existing_export_directory_is_left_untouched(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    audit = tmp_path / "audit.json"
    checkpoint.write_bytes(b"not needed because overwrite is rejected first")
    audit.write_text("{}")
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        export_bundle(checkpoint, audit, output)
    assert (output / "user-file").read_text() == "keep"


def continued_records():
    payload, audit = records(formal=True)
    payload["step"] = 43000
    audit["validation"]["checkpoints"][0]["step"] = 43000
    audit.update(training_complete_steps=100000, formal_50000_step_training=False,
                 formal_100000_step_training=True, run_sha256="continued-run")
    audit["continuation"] = {
        "schema_version": 1, "verified": True, "start_step": 50000, "end_step": 100000,
        "parent_run_target_steps": 50000, "parent_run_sha256": "parent-run", "parent_checkpoint_sha256": "parent-latest",
        "initialization": "resume_full_training_state", "root_initialization": "all_policy_parameters_random",
        "restoration": {"all_restored_exact": True, "first_batch_recomputed": True, "runtime_flags_restored": True},
        "metrics": {"exact_parent_prefix": True, "logged_step_coverage_verified": True},
        "final_optimizer": {"count": 50, "min": 100000.0, "max": 100000.0},
        "inherited_checkpoints": [{"sha256": "selected", "step": 43000}],
        "checkpoint_origins": [{"checkpoint_sha256": "selected", "step": 43000, "phase": "parent",
                                "origin_run_target_steps": 50000, "origin_run_sha256": "parent-run"}],
    }
    return payload, audit


def test_100k_run_exports_inherited_43k_best_without_relabelling_its_payload():
    payload, audit = continued_records()
    scope = verify_audit_claims(payload, audit, "selected")
    assert scope["role"] == "formal_100000_step_training_run"
    assert scope["checkpoint_training_step"] == 43000
    assert scope["checkpoint_origin_phase"] == "parent"
    assert scope["checkpoint_origin_run_target_steps"] == payload["config"]["steps"] == 50000
    assert scope["run_completed_steps"] == 100000


def test_100k_checkpoint_must_match_its_continuation_parent_identity():
    payload, audit = continued_records()
    payload.update(step=100000, initialization="resume_full_training_state", root_initialization="all_policy_parameters_random",
                   resume={"parent_run_sha256": "parent-run", "parent_checkpoint_sha256": "parent-latest"})
    payload["config"]["steps"] = 100000
    audit["validation"]["checkpoints"][0]["step"] = 100000
    audit["continuation"]["checkpoint_origins"] = [{"checkpoint_sha256": "selected", "step": 100000,
                                                  "phase": "continuation", "origin_run_target_steps": 100000,
                                                  "origin_run_sha256": "continued-run"}]
    assert verify_audit_claims(payload, audit, "selected")["checkpoint_origin_phase"] == "continuation"
    payload["resume"]["parent_checkpoint_sha256"] = "other-parent"
    with pytest.raises(ValueError, match="not bound"):
        verify_audit_claims(payload, audit, "selected")


@pytest.mark.parametrize("change", ["unverified", "wrong_parent", "missing_optimizer", "wrong_step", "wrong_formal", "wrong_config"])
def test_100k_export_rejects_fabricated_continuation_claims(change):
    payload, audit = continued_records()
    chain = audit["continuation"]
    if change == "unverified":
        chain["verified"] = False
    elif change == "wrong_parent":
        chain["parent_run_sha256"] = "different-run"
    elif change == "missing_optimizer":
        del chain["final_optimizer"]
    elif change == "wrong_step":
        chain["final_optimizer"]["min"] = 50000
    elif change == "wrong_formal":
        audit["formal_100000_step_training"] = False
    else:
        payload["config"]["steps"] = 100000
    with pytest.raises(ValueError):
        verify_audit_claims(payload, audit, "selected")
