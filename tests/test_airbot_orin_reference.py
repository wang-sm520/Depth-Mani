from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts import export_airbot_orin_reference as exporter


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rgb_for(index, frame, camera):
    shape = (4, 6, 3) if camera == exporter.CAMERA_KEYS[0] else (4, 4, 3)
    return np.full(shape, (index + frame + (20 if camera.endswith("wrist") else 0)) % 256, np.uint8)


def depth_for(images):
    result = np.ones((2, 2, 16, 16), dtype=np.float32)
    for i, rgb in enumerate(images):
        result[i, 0] = np.float32(rgb[0, 0, 0]) / np.float32(255)
    result[:, :, 0] = 0
    return result


class FakeDepthModel(torch.nn.Module):
    def forward(self, *, pixel_values):
        return pixel_values.mean(dim=1)


class FakeTransform:
    def __init__(self, config, provenance):
        self.config = config
        self._provenance = provenance
        self._model = FakeDepthModel()

    def provenance(self):
        return deepcopy(self._provenance)

    def __call__(self, images):
        for i, image in enumerate(images):
            pixel_values = torch.full((1, 3, 14, 14 * (i + 1)), float(image[0, 0, 0]) / 255)
            self._model(pixel_values=pixel_values)
        return depth_for(images)


class FakePolicy:
    def __init__(self, bundle, manifest):
        self.checkpoint_sha256 = exporter.sha256(bundle / "policy.pt")
        self.manifest = manifest
        self.camera_keys = list(exporter.CAMERA_KEYS)
        self.prompt = manifest["prompt"]
        self.depth_config = manifest["depth_config"]
        self.depth_provenance = manifest["depth_provenance"]
        self._depth_transform = None
        self.rgb_calls = 0

    def infer_rgb(self, state, rgb, prompt=None):
        self.rgb_calls += 1
        assert prompt == self.prompt
        if self._depth_transform is None:
            self._depth_transform = FakeTransform(self.depth_config, self.depth_provenance)
        return self.infer_depth(state, self._depth_transform([rgb[key] for key in self.camera_keys]), prompt)

    def infer_depth(self, state, depth, prompt=None):
        assert prompt == self.prompt
        return np.broadcast_to(state + np.mean(depth[:, 0]), (8, 7)).copy().astype(np.float32)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, bundle = tmp_path / "project", tmp_path / "bundle"
    data, source_root = tmp_path / "converted", tmp_path / "original"
    for path in (root, bundle, data, source_root):
        path.mkdir()
    runtime_files = ["airbot_depth/depth.py", "airbot_depth/model.py", "airbot_depth/policy.py",
                     "depth_policy/model.py"]
    for name in runtime_files + ["airbot_depth/lerobot.py"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((exporter.ROOT / name).read_bytes())
        if name in runtime_files:
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (source_root / "source.txt").write_text("original fixture data")
    source_file = {"path": "source.txt", "bytes": (source_root / "source.txt").stat().st_size,
                   "sha256": exporter.sha256(source_root / "source.txt")}
    episode_data, episodes, frames = {}, [], []
    for index in exporter.EPISODE_INDICES:
        state = (np.arange(35, dtype=np.float32).reshape(5, 7) + index) / np.float32(1000)
        stamps = np.arange(5, dtype=np.float64) / 25
        depths = np.stack([depth_for([rgb_for(index, frame, key) for key in exporter.CAMERA_KEYS])
                           for frame in range(5)])
        path = data / "episodes" / f"episode_{index:06d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as stream:
            stream.create_dataset("state", data=state)
            stream.create_dataset("timestamp", data=stamps)
            stream.create_dataset("depth", data=depths)
        split = "validation" if index == 5 else "train"
        episodes.append({"episode_index": index, "frames": 5, "split": split,
                         "path": path.relative_to(data).as_posix(), "sha256": exporter.sha256(path)})
        episode_data[index] = {"state": state, "timestamp": stamps}
        for frame in (0, 2, 4):
            inputs = {}
            for key in exporter.CAMERA_KEYS:
                record = exporter.array_record(key, rgb_for(index, frame, key))
                record.pop("key")
                inputs[key] = record
            frames.append({"episode_index": index, "frame_index": frame, "split": split,
                           "timestamp": float(stamps[frame]),
                           "camera_video_pts": {key: index * 10 + float(stamps[frame])
                                                for key in exporter.CAMERA_KEYS},
                           "camera_rgb_inputs": inputs, "mask_exact_match": True,
                           "same_rgb_inputs_reused_for_both_paths": True,
                           "depth_max_abs_difference": 0.0, "action_chunk_max_abs_difference": 0.0})
    manifest = {"episodes": episodes, "camera_keys": list(exporter.CAMERA_KEYS), "prompt": "pick the bag",
                "source": {"root": str(source_root), "files": [source_file]},
                "depth_config": {"image_size": 16}, "depth_provenance": {"original": "fixture"},
                "state_names": [f"joint{i}" for i in range(7)],
                "action_names": [f"joint{i}" for i in range(7)],
                "action_units": ["rad"] * 6 + ["m"], "action_semantics": "absolute_joint_position", "fps": 25}
    write_json(data / "manifest.json", manifest)
    original_sha = "a" * 64
    audit = {"schema_version": 1, "status": "passed", "formal_100000_step_training": True,
             "training_complete_steps": 100000, "data": str(data),
             "manifest_sha256": exporter.sha256(data / "manifest.json"),
             "roundtrip": {"checkpoint_sha256": original_sha, "passed": True,
                           "runtime_depth_provenance_exact_match": True, "camera_field_order_verified": True,
                           "camera_keys": list(exporter.CAMERA_KEYS),
                           "episode_indices": list(exporter.EPISODE_INDICES),
                           "absolute_tolerance": exporter.SOURCE_ATOL, "frames": frames}}
    write_json(tmp_path / "audit.json", audit)
    (bundle / "audit.json").write_bytes((tmp_path / "audit.json").read_bytes())
    (bundle / "policy.pt").write_bytes(b"fixture checkpoint; no real model is loaded")
    checkpoint_sha = exporter.sha256(bundle / "policy.pt")
    versions = {"python": "3.11 fixture", "machine": "x86_64", "packages": {"torch": "fixture"},
                "cuda_runtime": "12.8", "cudnn": 92000, "gpu": "reference GPU", "validation_device": "cuda"}
    write_json(bundle / "runtime-versions.json", versions)
    write_json(bundle / "bundle-validation.json", {"status": "passed", "exported_checkpoint_sha256": checkpoint_sha,
                                                    "source_checkpoint_sha256": original_sha})
    inventory = [{"path": path.relative_to(bundle).as_posix(), "bytes": path.stat().st_size,
                  "sha256": exporter.sha256(path)} for path in sorted(bundle.rglob("*")) if path.is_file()]
    write_json(bundle / "manifest.json", {
        "schema_version": 1, "status": "complete", "files": inventory,
        "scope": {"formal_training_run": True, "run_completed_steps": 100000, "checkpoint_training_step": 76000},
        "exported_checkpoint_sha256": checkpoint_sha, "source_checkpoint_sha256": original_sha,
        "audit_sha256": exporter.sha256(tmp_path / "audit.json"),
    })

    class FakeSource:
        camera_keys = list(exporter.CAMERA_KEYS)
        prompt = manifest["prompt"]
        info = {"features": {key: {"shape": list(rgb_for(0, 0, key).shape)} for key in exporter.CAMERA_KEYS}}

        def load_episode(self, index):
            return deepcopy(episode_data[index])

        def rgb_frames(self, index, key, stamps):
            return [(rgb_for(index, round(float(stamp) * 25), key), index * 10 + float(stamp)) for stamp in stamps]

        def fingerprint(self, indices):
            return {"root": str(source_root), "selected_episode_indices": indices,
                    "files": [{"path": "source.txt", "bytes": (source_root / "source.txt").stat().st_size,
                               "sha256": exporter.sha256(source_root / "source.txt")}]}

    source, policy = FakeSource(), FakePolicy(bundle, manifest)
    monkeypatch.setattr(exporter, "_load_source", lambda _: source)
    monkeypatch.setattr(exporter, "_load_runtime", lambda *_: (policy, versions))
    return SimpleNamespace(root=root, bundle=bundle, data=data, source_root=source_root,
                           source=source, policy=policy, versions=versions, manifest=manifest, audit=audit,
                           audit_path=tmp_path / "audit.json", output=tmp_path / "reference")


def run(fixture):
    return exporter.export_reference(fixture.bundle, fixture.audit_path, fixture.output, root=fixture.root)


def test_reference_reproduces_all_audited_inputs_without_claiming_target_validation(prepared):
    original_bundle = {path.relative_to(prepared.bundle): path.read_bytes()
                       for path in prepared.bundle.rglob("*") if path.is_file()}
    reference = run(prepared)
    assert reference["status"] == "complete" and reference["cross_platform_validated"] is False
    assert reference["robot_executed"] is False
    assert reference["checkpoint_training_step"] == 76000
    assert reference["tolerances"] == exporter.TOLERANCES
    assert reference["tolerances_fixed_before_target_measurement"] is True
    assert reference["checkpoint_sha256"] == exporter.sha256(prepared.bundle / "policy.pt")
    assert prepared.policy.rgb_calls == 12
    assert [(row["episode_index"], row["frame_index"], row["split"]) for row in reference["samples"]] == [
        (row["episode_index"], row["frame_index"], row["split"]) for row in prepared.audit["roundtrip"]["frames"]]
    for sample in reference["samples"]:
        path = prepared.output / sample["path"]
        assert exporter.sha256(path) == sample["sha256"]
        assert all(value == 0 for value in sample["source_reproduction"].values())
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == exporter.NPZ_KEYS
            assert arrays["depth"].dtype == np.float32
            assert arrays["processor_head"].shape == (1, 3, 14, 14)
            assert arrays["processor_wrist"].shape == (1, 3, 14, 28)
            for record in sample["arrays"]:
                assert exporter.array_record(record["key"], arrays[record["key"]]) == record
            assert np.array_equal(arrays["actions_rgb"], arrays["actions_depth"])
            assert float(arrays["processor_head"][0, 0, 0, 0]) == pytest.approx(
                float(arrays[exporter.CAMERA_KEYS[0]][0, 0, 0]) / 255)
    for line in (prepared.output / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert digest == exporter.sha256(prepared.output / relative)
    assert json.loads((prepared.output / "reference.json").read_text()) == reference
    assert original_bundle == {path.relative_to(prepared.bundle): path.read_bytes()
                               for path in prepared.bundle.rglob("*") if path.is_file()}
    assert not list(prepared.output.parent.glob(".reference.stage-*"))


def test_existing_output_is_untouched_before_loading_any_model(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("user file")
    with pytest.raises(FileExistsError):
        exporter.export_reference("missing bundle", "missing audit", output)
    assert (output / "keep").read_text() == "user file"


@pytest.mark.parametrize("change", ["bundle_weight", "audit", "data_manifest", "h5", "source", "runtime_source",
                                    "runtime_version", "bundle_output"])
def test_changed_identity_or_environment_is_rejected_without_publishing(prepared, change):
    if change == "bundle_weight":
        (prepared.bundle / "policy.pt").write_bytes(b"different")
    elif change == "audit":
        prepared.audit_path.write_text("{}")
    elif change == "data_manifest":
        (prepared.data / "manifest.json").write_text("{}")
    elif change == "h5":
        with h5py.File(prepared.data / prepared.manifest["episodes"][0]["path"], "r+") as stream:
            stream["state"][0, 0] = 100
    elif change == "source":
        (prepared.source_root / "source.txt").write_text("modified original data")
    elif change == "runtime_source":
        (prepared.root / "airbot_depth/depth.py").write_text("# drift\n")
    elif change == "runtime_version":
        prepared.versions["packages"]["torch"] = "changed-target-version"
    else:
        prepared.output = prepared.bundle / "new-reference"
    with pytest.raises(ValueError):
        run(prepared)
    assert not prepared.output.exists()
    assert not list(prepared.output.parent.glob(f".{prepared.output.name}.stage-*"))


@pytest.mark.parametrize("change", ["missing", "reordered", "duplicate", "split", "tolerance", "mask", "error"])
def test_reference_selection_cannot_drop_reorder_or_replace_audited_frames(prepared, change):
    audit = deepcopy(prepared.audit)
    frames = audit["roundtrip"]["frames"]
    if change == "missing":
        frames.pop()
    elif change == "reordered":
        frames.reverse()
    elif change == "duplicate":
        frames[1] = deepcopy(frames[0])
    elif change == "split":
        frames[-1]["split"] = "train"
    elif change == "tolerance":
        audit["roundtrip"]["absolute_tolerance"] = 0.1
    elif change == "mask":
        frames[0]["mask_exact_match"] = False
    else:
        frames[0]["depth_max_abs_difference"] = float("nan")
    with pytest.raises(ValueError):
        exporter.selected_frames(prepared.manifest, audit)


@pytest.mark.parametrize("change", ["rgb", "pts", "state", "timestamp"])
def test_source_decoder_must_reproduce_the_original_audit_and_state(prepared, monkeypatch, change):
    if change in {"rgb", "pts"}:
        original = prepared.source.rgb_frames

        def changed(index, key, stamps):
            rows = original(index, key, stamps)
            rgb, pts = rows[0]
            if change == "rgb":
                rgb = rgb.copy()
                rgb[0, 0, 0] ^= 1
            else:
                pts += 0.01
            rows[0] = (rgb, pts)
            return rows

        monkeypatch.setattr(prepared.source, "rgb_frames", changed)
    else:
        original = prepared.source.load_episode

        def changed(index):
            result = original(index)
            result[change][0] += 0.5
            return result

        monkeypatch.setattr(prepared.source, "load_episode", changed)
    with pytest.raises(ValueError):
        run(prepared)
    assert not prepared.output.exists()
    assert not list(prepared.output.parent.glob(".reference.stage-*"))


def test_captured_processor_is_actual_forward_input_and_hook_is_removed_on_error():
    class BrokenTransform:
        def __init__(self):
            self._model = FakeDepthModel()

        def __call__(self, _images):
            self._model(pixel_values=torch.ones((1, 3, 14, 14), dtype=torch.float16))

    transform = BrokenTransform()
    with pytest.raises(ValueError, match="actual Depth Anything processor"):
        exporter.capture_depth(transform, [rgb_for(0, 0, key) for key in exporter.CAMERA_KEYS])
    assert not transform._model._forward_pre_hooks


@pytest.mark.parametrize("change", ["depth", "mask", "nonfinite"])
def test_reference_rejects_source_depth_drift_before_any_target_comparison(prepared, change):
    rgb = {key: rgb_for(0, 0, key) for key in exporter.CAMERA_KEYS}
    cached = depth_for(list(rgb.values()))
    if change == "depth":
        cached[0, 0, 5, 5] += 0.1
    elif change == "mask":
        cached[0, 1, 5, 5] = 0
    else:
        cached[0, 0, 5, 5] = float("nan")
    with pytest.raises(ValueError):
        exporter.infer_reference(prepared.policy, np.zeros(7, dtype=np.float32), rgb, cached)


def test_inference_failure_never_publishes_a_partial_reference(prepared, monkeypatch):
    original = prepared.policy.infer_rgb

    def failed(state, rgb, prompt=None):
        if prepared.policy.rgb_calls == 1:
            raise RuntimeError("fixture inference failed")
        return original(state, rgb, prompt)

    monkeypatch.setattr(prepared.policy, "infer_rgb", failed)
    with pytest.raises(RuntimeError, match="fixture inference failed"):
        run(prepared)
    assert not prepared.output.exists()
    assert not list(prepared.output.parent.glob(".reference.stage-*"))
