from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airbot_depth.depth import DEFAULT_DEPTH_CONFIG
from airbot_deploy.client import validate_metadata
from airbot_orin import runtime


class InputModel(torch.nn.Module):
    def forward(self, *, pixel_values):
        return pixel_values.mean(dim=1)


class TraceTransform:
    def __init__(self, depth, processors):
        self.depth, self.processors = depth, processors
        self._model = InputModel()
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        assert len(images) == 2
        for values in self.processors:
            self._model(pixel_values=torch.from_numpy(values))
        return self.depth.copy()


class SequencePolicy:
    prompt = "pick the bag"

    def __init__(self, actions=None):
        self.actions = actions or [np.zeros((8, 7), np.float32), np.zeros((8, 7), np.float32)]
        self.calls = 0

    def infer_depth(self, state, depth, prompt=None):
        assert prompt == self.prompt and state.shape == (7,)
        value = self.actions[self.calls % len(self.actions)]
        self.calls += 1
        return value.copy()


@pytest.fixture
def sample():
    depth = np.ones((2, 2, 128, 128), dtype=np.float32)
    depth[:, 0] = 0.5
    depth[:, :, :10] = 0
    return {
        "state": np.zeros(7, np.float32),
        "observation.images.head": np.zeros((1080, 1920, 3), np.uint8),
        "observation.images.wrist": np.zeros((480, 848, 3), np.uint8),
        "depth": depth, "actions_depth": np.zeros((8, 7), np.float32),
        "actions_rgb": np.zeros((8, 7), np.float32),
        "processor_head": np.zeros((1, 3, 14, 28), np.float32),
        "processor_wrist": np.zeros((1, 3, 14, 14), np.float32),
    }


def components(sample):
    return SequencePolicy(), TraceTransform(sample["depth"].copy(),
                                            [sample["processor_head"].copy(), sample["processor_wrist"].copy()])


def test_zero_difference_passes_all_layers_using_actual_captured_forward_inputs(sample):
    policy, transform = components(sample)
    result = runtime.compare_sample(policy, transform, sample)
    assert result["passed"] is True and result["mask_exact"] is True
    assert all(item == {"max_abs": 0.0, "mean_abs": 0.0} for item in result["processor"])
    assert result["depth_valid_pixels"] == {"max_abs": 0.0, "mean_abs": 0.0}
    assert result["actions_refdepth"] == result["actions_rgb"] == {"joints_max_abs_rad": 0.0, "gripper_max_abs_m": 0.0}
    assert policy.calls == 2 and transform.calls == 1
    assert not transform._model._forward_pre_hooks


@pytest.mark.parametrize("failure", ["processor", "depth_peak", "depth_mean", "mask", "policy_joints",
                                     "policy_gripper", "rgb_joints", "rgb_gripper"])
def test_each_fixed_gate_independently_rejects_excess_error(sample, failure):
    policy, transform = components(sample)
    limits = runtime.TOLERANCES
    if failure == "processor":
        transform.processors[0][0, 0, 0, 0] = limits["processor_atol"] * 1.01
    elif failure == "depth_peak":
        transform.depth[0, 0, 20, 20] += limits["depth_max_abs"] * 1.01
    elif failure == "depth_mean":
        valid = transform.depth[:, 1].astype(bool)
        transform.depth[:, 0][valid] += limits["depth_mean_abs"] * 1.01
    elif failure == "mask":
        transform.depth[0, 1, 20, 20] = 0
        transform.depth[0, 0, 20, 20] = 0
    else:
        isolated = failure.startswith("policy")
        gripper = failure.endswith("gripper")
        prefix = "actions_refdepth" if isolated else "actions_rgb"
        key = prefix + ("_gripper_atol" if gripper else "_joints_atol")
        policy.actions[0 if isolated else 1][0, 6 if gripper else 0] = limits[key] * 1.01
    result = runtime.compare_sample(policy, transform, sample)
    assert result["passed"] is False


def test_within_threshold_errors_pass_without_modifying_recorded_thresholds(sample):
    policy, transform = components(sample)
    limits = deepcopy(runtime.TOLERANCES)
    transform.processors[1][0, 0, 0, 0] = limits["processor_atol"] * 0.9
    valid = transform.depth[:, 1].astype(bool)
    transform.depth[:, 0][valid] += limits["depth_mean_abs"] * 0.9
    policy.actions[0][:, :6] = limits["actions_refdepth_joints_atol"] * 0.9
    policy.actions[0][:, 6] = limits["actions_refdepth_gripper_atol"] * 0.9
    policy.actions[1][:, :6] = limits["actions_rgb_joints_atol"] * 0.9
    policy.actions[1][:, 6] = limits["actions_rgb_gripper_atol"] * 0.9
    assert runtime.compare_sample(policy, transform, sample)["passed"] is True
    assert runtime.TOLERANCES == limits


@pytest.mark.parametrize("failure", ["processor_shape", "processor_nonfinite", "depth_shape", "depth_nonfinite",
                                     "action_shape", "action_nonfinite"])
def test_invalid_tensor_geometry_and_nonfinite_values_never_produce_a_pass(sample, failure):
    policy, transform = components(sample)
    if failure == "processor_shape":
        transform.processors[0] = transform.processors[0][..., :-1]
    elif failure == "processor_nonfinite":
        transform.processors[0][0, 0, 0, 0] = np.nan
    elif failure == "depth_shape":
        transform.depth = transform.depth[..., :-1]
    elif failure == "depth_nonfinite":
        transform.depth[0, 0, 20, 20] = np.inf
    elif failure == "action_shape":
        policy.actions[0] = np.zeros((4, 7), np.float32)
    else:
        policy.actions[1][0, 0] = np.nan
    with pytest.raises(ValueError):
        runtime.compare_sample(policy, transform, sample)
    assert not transform._model._forward_pre_hooks


@pytest.mark.parametrize("change", ["implementation_sha256", "config", "processor", "files_sha256"])
def test_cross_platform_artifact_contract_allows_only_library_versions_to_differ(change):
    original = {"schema_version": 1, "implementation_sha256": "a" * 64,
                "config": {"representation": "relative_inverse_depth"},
                "processor": {"use_fast": False}, "files_sha256": {"model.safetensors": "b" * 64},
                "libraries": {"torch": "original-CUDA"}}
    target = deepcopy(original)
    target["libraries"]["torch"] = "native-Jetson-CUDA"
    assert runtime.artifact_provenance_matches(original, target)
    target[change] = "changed"
    assert not runtime.artifact_provenance_matches(original, target)
    assert original["libraries"]["torch"] == "original-CUDA"


@pytest.fixture
def attested(tmp_path, monkeypatch):
    root = tmp_path / "package"
    (root / "references").mkdir(parents=True)
    (root / "runtime").mkdir()
    (root / "manifest.json").write_text("fixture immutable package manifest")
    samples = [{"path": f"samples/{index}.npz", "sha256": f"{index:064x}"} for index in range(12)]
    reference = {"checkpoint_sha256": "c" * 64, "samples": samples}
    (root / "references/reference.json").write_text(json.dumps(reference))
    original = {"schema_version": 1, "libraries": {"torch": "original"}, "config": {"frozen": True}}
    target = {**deepcopy(original), "libraries": {"torch": "native"}}
    policy = SimpleNamespace(depth_provenance=original)
    environment = {"torch": "native", "threads": 4, "device": "cpu"}
    monkeypatch.setattr(runtime, "verify_package", lambda _: {"status": "complete"})
    monkeypatch.setattr(runtime, "configure_runtime", lambda *_: None)
    monkeypatch.setattr(runtime, "load_reference", lambda _: reference)
    monkeypatch.setattr(runtime, "load_components", lambda *_: (policy, object(), target))
    monkeypatch.setattr(runtime, "environment", lambda _: deepcopy(environment))
    zero = {"max_abs": 0.0, "mean_abs": 0.0}
    actions = {"joints_max_abs_rad": 0.0, "gripper_max_abs_m": 0.0}
    comparisons = [{**sample, "passed": True, "processor": [deepcopy(zero), deepcopy(zero)],
                    "mask_exact": True, "depth_valid_pixels": deepcopy(zero),
                    "actions_refdepth": deepcopy(actions), "actions_rgb": deepcopy(actions)} for sample in samples]
    report = {"schema_version": 1, "status": "passed", "checkpoint_sha256": reference["checkpoint_sha256"],
              "package_manifest_sha256": runtime.sha256(root / "manifest.json"),
              "reference_sha256": runtime.sha256(root / "references/reference.json"),
              "runtime_source_sha256": runtime.sha256(runtime.__file__), "environment": deepcopy(environment),
              "tolerances": deepcopy(runtime.TOLERANCES), "robot_executed": False,
              "training_depth_provenance": deepcopy(original), "runtime_depth_provenance": deepcopy(target),
              "comparisons": comparisons}
    path = root / "runtime/attestation.json"
    path.write_text(json.dumps(report))
    return SimpleNamespace(root=root, reference=reference, policy=policy, target=target,
                           environment=environment, report=report, path=path)


def test_matching_attestation_retains_original_and_target_provenance_separately(attested):
    policy, _, report = runtime.verify_attestation(attested.root, attested.path, "cpu", 4)
    assert policy is attested.policy
    assert report["training_depth_provenance"]["libraries"]["torch"] == "original"
    assert report["runtime_depth_provenance"]["libraries"]["torch"] == "native"


@pytest.mark.parametrize("field", ["checkpoint_sha256", "package_manifest_sha256", "reference_sha256",
                                   "runtime_source_sha256", "environment", "status", "tolerances",
                                   "training_depth_provenance", "runtime_depth_provenance"])
def test_stale_attestation_is_rejected_before_serving(attested, field):
    attested.report[field] = "changed"
    attested.path.write_text(json.dumps(attested.report))
    with pytest.raises(ValueError):
        runtime.verify_attestation(attested.root, attested.path, "cpu", 4)


@pytest.mark.parametrize("change", ["missing_sample", "sample_identity", "false", "failed_metric", "nan_metric"])
def test_attestation_must_cover_every_sample_and_reapply_the_fixed_numeric_gates(attested, change):
    if change == "missing_sample":
        attested.report["comparisons"].pop()
    elif change == "sample_identity":
        attested.report["comparisons"][0]["sha256"] = "e" * 64
    elif change == "false":
        attested.report["comparisons"][0]["passed"] = False
    else:
        attested.report["comparisons"][0]["actions_rgb"]["joints_max_abs_rad"] = (
            0.1 if change == "failed_metric" else float("nan"))
    attested.path.write_text(json.dumps(attested.report))
    with pytest.raises(ValueError):
        runtime.verify_attestation(attested.root, attested.path, "cpu", 4)


def test_changed_target_runtime_provenance_invalidates_a_previously_matching_attestation(attested):
    attested.target["libraries"]["torch"] = "different native build"
    with pytest.raises(ValueError, match="changed since validation"):
        runtime.verify_attestation(attested.root, attested.path, "cpu", 4)


def contract_policy():
    config = deepcopy(DEFAULT_DEPTH_CONFIG)
    provenance = {"config": config, "files_sha256": {key: "a" * 64 for key in (
        "config.json", "preprocessor_config.json", "model.safetensors")},
        "implementation_sha256": "b" * 64, "libraries": {"torch": "training"},
        "processor": {"class": "DPTImageProcessor"}}
    names, units = [f"joint{i}.pos" for i in range(1, 7)] + ["eef.pos"], ["rad"] * 6 + ["m"]
    manifest = {"state_dim": 7, "action_dim": 7, "state_names": names, "action_names": names,
                "state_units": units, "action_units": units, "action_semantics": "absolute_joint_position",
                "depth_config": config, "depth_provenance": provenance,
                "model_input_quantization": "float32_to_float16_to_float32"}

    class Policy:
        camera_keys = list(runtime.CAMERA_KEYS)
        prompt, fps = "pick the bag", 25.0
        model = SimpleNamespace(horizon=8)
        checkpoint_sha256 = "c" * 64
        depth_provenance = provenance
        calls = 0

        def _validate_state_and_prompt(self, state, prompt):
            if np.asarray(state).shape != (7,) or prompt != self.prompt:
                raise ValueError("Invalid source state/prompt")

        def infer_depth(self, state, depth, prompt=None):
            self._validate_state_and_prompt(state, prompt)
            self.calls += 1
            return np.zeros((8, 7), np.float32)

        def infer_rgb(self, *args, **kwargs):
            raise AssertionError("Do not override, call, or weaken original strict RGB provenance check")

    policy = Policy()
    policy.manifest = manifest
    return policy


def test_separate_rgb_wrapper_preserves_policy_validation_and_frozen_training_provenance(sample):
    policy = contract_policy()
    _, transform = components(sample)
    wrapper = runtime.AttestedPolicy(policy, transform)
    before = deepcopy(policy.depth_provenance)
    rgb = {key: sample[key] for key in runtime.CAMERA_KEYS}
    actions = wrapper.infer_rgb(sample["state"], rgb, prompt=policy.prompt)
    assert actions.shape == (8, 7) and transform.calls == policy.calls == 1
    assert policy.depth_provenance == before
    with pytest.raises(ValueError, match="state/prompt"):
        wrapper.infer_rgb(sample["state"], rgb, prompt="different prompt")
    assert transform.calls == 1


def test_separate_rgb_wrapper_rejects_client_side_224_resize(sample):
    policy = contract_policy()
    _, transform = components(sample)
    wrapper = runtime.AttestedPolicy(policy, transform)
    images = {key: np.zeros((224, 224, 3), np.uint8) for key in runtime.CAMERA_KEYS}
    with pytest.raises(ValueError):
        wrapper.infer_rgb(sample["state"], images, prompt=policy.prompt)
    assert transform.calls == 0


def test_extra_runtime_metadata_matches_existing_robot_client_without_replacing_training_provenance(tmp_path):
    policy = contract_policy()
    path = tmp_path / "attestation.json"
    runtime_provenance = deepcopy(policy.depth_provenance)
    runtime_provenance["libraries"]["torch"] = "target"
    attestation = {"runtime_depth_provenance": runtime_provenance}
    path.write_text(json.dumps(attestation))
    adapter = runtime.AttestedAdapter(policy, path, attestation)
    metadata = adapter.metadata
    base_fields = {"schema_version", "policy_type", "protocol", "preprocessing", "max_message_bytes"}
    profile = {"policy": {key: value for key, value in metadata.items() if key not in base_fields},
               "transport": {"max_message_bytes": metadata["max_message_bytes"]}}
    assert validate_metadata(metadata, profile) == metadata
    assert metadata["runtime_attestation_sha256"] == runtime.sha256(path)
    assert metadata["preprocessing"]["depth_provenance"]["libraries"]["torch"] == "training"
    assert metadata["runtime_depth_provenance"]["libraries"]["torch"] == "target"
    profile["policy"]["runtime_attestation_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime_attestation_sha256"):
        validate_metadata(metadata, profile)


@pytest.fixture
def verified_package(tmp_path, monkeypatch):
    root = tmp_path / "verified-package"
    source_names = ["airbot_depth.depth", "airbot_depth.policy", "airbot_depth.model", "airbot_depth.serve",
                    "depth_policy.data", "depth_policy.model"]
    source_hashes = {}
    for name in source_names:
        relative = name.replace(".", "/") + ".py"
        path = root / "model" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen test source for {name}\n")
        source_hashes[relative] = runtime.sha256(path)
        monkeypatch.setattr(runtime.sys.modules[name], "__file__", str(path))
    (root / "model/policy.pt").write_bytes(b"unchanged model fixture")
    model = {"files": [{"path": path, "sha256": digest} for path, digest in source_hashes.items()]}
    (root / "model/manifest.json").write_text(json.dumps(model))
    (root / "airbot_orin").mkdir()
    runtime_path = root / "airbot_orin/runtime.py"
    runtime_path.write_text("# runtime assembly fixture\n")
    monkeypatch.setattr(runtime, "__file__", str(runtime_path))
    (root / "deploy.py").write_text("# launcher assembly fixture\n")
    (root / "references/samples").mkdir(parents=True)
    samples = []
    for episode, indices in ((0, [0, 117, 233]), (100, [0, 129, 258]), (199, [0, 90, 180]), (5, [0, 124, 247])):
        for index in indices:
            relative = f"samples/{episode}-{index}.npz"
            # File identity tests never run inference or deserialize this file.
            (root / "references" / relative).write_bytes(f"opaque {episode}/{index}".encode())
            samples.append({"episode_index": episode, "frame_index": index,
                            "split": "validation" if episode == 5 else "train",
                            "path": relative, "sha256": runtime.sha256(root / "references" / relative)})
    reference = {"schema_version": 1, "status": "complete", "role": "source_runtime_reference",
                 "camera_keys": list(runtime.CAMERA_KEYS), "tolerances": deepcopy(runtime.TOLERANCES),
                 "samples": samples, "checkpoint_sha256": runtime.sha256(root / "model/policy.pt"),
                 "bundle_manifest_sha256": runtime.sha256(root / "model/manifest.json"),
                 "source_files_sha256": source_hashes}
    reference_path = root / "references/reference.json"
    reference_path.write_text(json.dumps(reference))
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
              "sha256": runtime.sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "status": "complete", "files": files}))
    return SimpleNamespace(root=root, reference=reference, reference_path=reference_path)


def test_verified_package_binds_both_frozen_file_bytes_and_actual_import_locations(verified_package):
    assert runtime.verify_package(verified_package.root)["status"] == "complete"
    assert runtime.load_reference(verified_package.root) == verified_package.reference


@pytest.mark.parametrize("change", ["runtime_origin", "model_import_origin", "model_file_bytes"])
def test_unchanged_inventory_does_not_allow_runtime_imports_from_another_checkout(verified_package, monkeypatch, change):
    root = verified_package.root
    if change == "runtime_origin":
        monkeypatch.setattr(runtime, "__file__", str(root.parent / "other-runtime.py"))
    elif change == "model_import_origin":
        monkeypatch.setattr(runtime.sys.modules["airbot_depth.policy"], "__file__", str(root.parent / "other-policy.py"))
    else:
        (root / "model/airbot_depth/policy.py").write_text("# changed after package assembly\n")
    with pytest.raises(ValueError):
        runtime.verify_package(root)


@pytest.mark.parametrize("change", ["source_fingerprint", "checkpoint", "bundle_manifest", "sample_bytes",
                                    "relaxed_threshold", "source_bytes", "escaping_sample", "missing_sample"])
def test_reference_checks_bind_samples_sources_checkpoint_and_fixed_protocol(verified_package, change):
    root, reference = verified_package.root, verified_package.reference
    if change == "source_fingerprint":
        reference["source_files_sha256"]["airbot_depth/model.py"] = "f" * 64
    elif change == "checkpoint":
        reference["checkpoint_sha256"] = "f" * 64
    elif change == "bundle_manifest":
        reference["bundle_manifest_sha256"] = "f" * 64
    elif change == "sample_bytes":
        (root / "references" / reference["samples"][0]["path"]).write_bytes(b"modified")
    elif change == "relaxed_threshold":
        reference["tolerances"]["processor_atol"] = 0.5
    elif change == "source_bytes":
        (root / "model/airbot_depth/depth.py").write_text("# wrong bytes\n")
    elif change == "escaping_sample":
        reference["samples"][0]["path"] = "../../outside.npz"
    else:
        reference["samples"].pop()
    verified_package.reference_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        runtime.load_reference(root)


def test_server_recomputes_startup_sample_and_refuses_numerical_drift_before_listening(tmp_path, sample, monkeypatch):
    policy = contract_policy()
    _, transform = components(sample)
    path = tmp_path / "references/samples/first.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, **sample)
    monkeypatch.setattr(runtime, "verify_attestation", lambda *_: (policy, transform, {}))
    monkeypatch.setattr(runtime, "load_reference", lambda _: {"samples": [{"path": "samples/first.npz"}]})
    observed = []

    def failed_comparison(actual_policy, actual_transform, fixture):
        assert actual_policy is policy and actual_transform is transform
        assert set(fixture) == set(sample)
        observed.append(fixture["state"])
        return {"passed": False}

    def must_not_listen(*args, **kwargs):
        raise AssertionError("A stale numerical runtime must be rejected before creating the server")

    monkeypatch.setattr(runtime, "compare_sample", failed_comparison)
    monkeypatch.setattr(runtime, "AirbotWebsocketServer", must_not_listen)
    with pytest.raises(ValueError, match="Startup reference comparison failed"):
        runtime.serve(tmp_path, tmp_path / "attestation.json", device="cpu")
    assert len(observed) == 1
