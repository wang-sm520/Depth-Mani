"""Verify Orin artifact assembly without importing models or contacting hardware."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import export_airbot_orin as exporter


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _json(path, value):
    _write(path, json.dumps(value))


def _inventory(directory):
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
             "sha256": exporter.sha256(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"]


def _sums(directory, name="SHA256SUMS"):
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.relative_to(directory).as_posix() not in {name, "manifest.json"}:
            lines.append(f"{exporter.sha256(path)}  {path.relative_to(directory).as_posix()}\n")
    _write(directory / name, "".join(lines))


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "project"
    model = root / exporter.MODEL_DIRECTORY
    robot = root / exporter.ROBOT_DIRECTORY
    references = tmp_path / "references"
    client = tmp_path / "openpi-client"
    output = tmp_path / "orin-package"
    for relative in exporter.TEMPLATES:
        _write(root / relative, "raise RuntimeError('assembly must never execute this source')\n")
    _write(root / "airbot_orin/__init__.py", "raise RuntimeError('do not import runtime')\n")
    _write(root / "airbot_orin/validate.py", "raise RuntimeError('do not run inference')\n")
    _write(root / "airbot_orin/notes.txt", "not a runtime module")
    _write(root / ".venv/private-dependency", "do not copy an x86 environment")
    _write(client / "__init__.py", '__version__ = "0.1.0"\n')
    _write(client / "msgpack_numpy.py", "raise RuntimeError('do not import message codec')\n")
    _write(client / "websocket_client_policy.py", "not selected for vendoring")
    _write(model / "policy.pt", "opaque fixed policy bytes")
    checkpoint = exporter.sha256(model / "policy.pt")
    # The production CLI has no checkpoint override. Synthetic fixtures replace
    # its constant so the assembly gates can be exercised without GPU weights.
    monkeypatch.setattr(exporter, "POLICY_SHA256", checkpoint)
    audited = [{"episode_index": episode, "frame_index": frame, "split": "validation" if episode == 5 else "train",
                "timestamp": frame / 25, "camera_video_pts": {"head": frame / 25, "wrist": frame / 25}}
               for episode, frames in [(0, [0, 117, 233]), (100, [0, 129, 258]),
                                       (199, [0, 90, 180]), (5, [0, 124, 247])]
               for frame in frames]
    _json(model / "audit.json", {"manifest_sha256": "d" * 64, "roundtrip": {"frames": audited}})
    _json(model / "bundle-validation.json", {"status": "passed", "robot_executed": False})
    for name in ("policy.py", "depth.py"):
        _write(model / "airbot_depth" / name, "raise RuntimeError('must not import the model')\n")
    model_manifest = {"schema_version": 1, "status": "complete", "exported_checkpoint_sha256": checkpoint,
                      "audit_sha256": exporter.sha256(model / "audit.json"), "files": _inventory(model),
                      "scope": {"formal_training_run": True, "run_completed_steps": 100000,
                                "checkpoint_training_step": 76000}}
    _json(model / "manifest.json", model_manifest)
    _write(robot / "airbot_deploy/client.py", "raise RuntimeError('must not import robot code')\n")
    _write(robot / "airbot_deploy/__init__.py", "# client\n")
    _write(robot / "configs/airbot_paperbag_deploy.json", "{\"client_only\": true}\n")
    _sums(robot)
    robot_manifest = {"schema_version": 1, "status": "complete", "bundle_type": "airbot_robot_client",
                      "client_only": True, "policy_checkpoint_sha256": checkpoint, "files": _inventory(robot),
                      "source_model": {"manifest_sha256": exporter.sha256(model / "manifest.json"),
                                       "validation_sha256": exporter.sha256(model / "bundle-validation.json")}}
    _json(robot / "manifest.json", robot_manifest)
    samples = []
    for frame in audited:
        relative = f"samples/episode_{frame['episode_index']:06d}_frame_{frame['frame_index']:06d}.npz"
        _write(references / relative, json.dumps(frame))
        samples.append({**deepcopy(frame), "path": relative, "sha256": exporter.sha256(references / relative)})
    reference = {"schema_version": 1, "status": "complete", "role": "source_runtime_reference",
                 "cross_platform_validated": False, "checkpoint_sha256": checkpoint,
                 "bundle_manifest_sha256": exporter.sha256(model / "manifest.json"),
                 "audit_sha256": exporter.sha256(model / "audit.json"), "dataset_manifest_sha256": "d" * 64,
                 "samples": samples}
    _json(references / "reference.json", reference)
    _sums(references)
    return {"root": root, "model": model, "robot": robot, "references": references, "reference": reference,
            "client": client, "output": output, "checkpoint": checkpoint}


def _export(sources, output=None):
    return exporter.export_orin(output or sources["output"], sources["references"], root=sources["root"],
                                client_source=sources["client"])


def test_assembly_preserves_all_source_bytes_and_does_not_claim_native_readiness(sources):
    originals = {directory: {path.relative_to(directory).as_posix(): path.read_bytes()
                            for path in directory.rglob("*") if path.is_file()}
                 for directory in (sources["model"], sources["robot"], sources["references"])}
    manifest = _export(sources)
    output = sources["output"]
    assert manifest["status"] == "complete"
    assert manifest["state"] == "awaiting_orin_installation"
    assert manifest["cross_platform_validated"] is False
    assert manifest["inference_ready"] is False
    assert manifest["robot_executed"] is False
    assert manifest["environment_bundled"] is False
    assert manifest["policy_checkpoint_sha256"] == sources["checkpoint"]
    assert manifest["source_identity"]["references"]["sample_count"] == 12
    for name in ("model", "robot", "references"):
        directory = sources[name]
        for relative, value in originals[directory].items():
            assert (output / name / relative).read_bytes() == value
            assert (directory / relative).read_bytes() == value
    inventory = {item["path"]: item for item in manifest["files"]}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert actual == set(inventory) | {"manifest.json"}
    assert {"deploy.py", "prepare_env.py", "README.md", "airbot_orin/validate.py",
            "vendor/openpi_client/__init__.py", "vendor/openpi_client/msgpack_numpy.py"} <= actual
    assert not any(path.startswith("runtime/") or ".venv" in path or path.endswith("notes.txt") for path in actual)
    assert not any(path.endswith("websocket_client_policy.py") for path in actual)
    assert not any(path.is_symlink() for path in output.rglob("*"))
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    exporter.verify_bundle(output)
    assert not list(output.parent.glob(f".{output.name}.stage-*"))


@pytest.mark.parametrize("kind", ["model", "robot", "reference_sample"])
def test_changed_source_payload_is_rejected_before_any_package_is_published(sources, kind):
    path = {"model": sources["model"] / "policy.pt", "robot": sources["robot"] / "airbot_deploy/client.py",
            "reference_sample": sources["references"] / sources["reference"]["samples"][0]["path"]}[kind]
    path.write_text("tampered after its source audit")
    with pytest.raises(ValueError, match="checksum"):
        _export(sources)
    assert not sources["output"].exists()
    assert not list(sources["output"].parent.glob(f".{sources['output'].name}.stage-*"))


@pytest.mark.parametrize("change", ["wrong_checkpoint", "missing_sample", "duplicate_sample", "wrong_frame", "claimed_native_validation"])
def test_reference_set_must_match_the_full_fixed_source_audit(sources, change):
    reference = deepcopy(sources["reference"])
    if change == "wrong_checkpoint":
        reference["checkpoint_sha256"] = "f" * 64
    elif change == "missing_sample":
        reference["samples"].pop()
    elif change == "duplicate_sample":
        reference["samples"][1] = deepcopy(reference["samples"][0])
    elif change == "wrong_frame":
        reference["samples"][0]["frame_index"] = 999
    else:
        reference["cross_platform_validated"] = True
    _json(sources["references"] / "reference.json", reference)
    _sums(sources["references"])
    with pytest.raises(ValueError):
        _export(sources)
    assert not sources["output"].exists()


@pytest.mark.parametrize("kind", ["existing", "inside_model", "inside_robot", "inside_references"])
def test_output_never_overwrites_or_enters_an_immutable_source(sources, kind):
    if kind == "existing":
        output = sources["output"]
        output.mkdir()
        (output / "keep").write_text("existing user data")
        expected = FileExistsError
    else:
        output = sources[kind.removeprefix("inside_")] / "new-output"
        expected = ValueError
    with pytest.raises(expected):
        _export(sources, output)
    if kind == "existing":
        assert (output / "keep").read_text() == "existing user data"
    else:
        assert not output.exists()


@pytest.mark.parametrize("problem", ["unlisted_file", "unsafe_path", "symlink", "unsafe_client_init"])
def test_only_closed_verified_file_sets_and_inert_client_initializer_are_accepted(sources, problem):
    if problem == "unlisted_file":
        _write(sources["model"] / "untracked.py", "do not quietly include or ignore this")
    elif problem == "unsafe_path":
        reference = deepcopy(sources["reference"])
        reference["samples"][0]["path"] = "../outside.npz"
        _json(sources["references"] / "reference.json", reference)
        _sums(sources["references"])
    elif problem == "symlink":
        original = sources["robot"] / "airbot_deploy/client.py"
        outside = sources["root"].parent / "outside.py"
        outside.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(outside)
    else:
        _write(sources["client"] / "__init__.py", "import unexpected_client_dependency\n")
    with pytest.raises(ValueError):
        _export(sources)
    assert not sources["output"].exists()


def test_source_change_during_copy_cleans_stage_and_never_publishes(sources, monkeypatch):
    actual_copy = exporter._copy
    changed = sources["root"] / "scripts/deploy_airbot_orin.py"

    def copy_then_modify(source, destination, digest):
        actual_copy(source, destination, digest)
        if source == changed:
            changed.write_text("modified during assembly")

    monkeypatch.setattr(exporter, "_copy", copy_then_modify)
    with pytest.raises(ValueError, match="changed during"):
        _export(sources)
    assert not sources["output"].exists()
    assert not list(sources["output"].parent.glob(f".{sources['output'].name}.stage-*"))


def test_racing_destination_is_preserved_and_stage_is_cleaned(sources, monkeypatch):
    publish = exporter.publish_new

    def race(stage, output):
        output.mkdir()
        (output / "keep").write_text("another publisher won")
        publish(stage, output)

    monkeypatch.setattr(exporter, "publish_new", race)
    with pytest.raises(FileExistsError):
        _export(sources)
    assert (sources["output"] / "keep").read_text() == "another publisher won"
    assert not list(sources["output"].parent.glob(f".{sources['output'].name}.stage-*"))


def test_cli_requires_reference_directory_and_help_runs_outside_repo(tmp_path):
    with pytest.raises(SystemExit) as error:
        exporter.main(["--output", str(tmp_path / "must-not-create")])
    assert error.value.code == 2
    assert not (tmp_path / "must-not-create").exists()
    script = Path(exporter.__file__).resolve()
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10, check=True)
    assert "--references" in result.stdout and "--output" in result.stdout


def test_selected_robot_snapshot_is_verified_and_copied_without_the_old_snapshot(sources):
    selected = sources["root"] / "native-robot-client"
    sources["robot"].rename(selected)
    manifest = exporter.export_orin(sources["output"], sources["references"], root=sources["root"],
                                    client_source=sources["client"], robot_bundle=selected)
    assert manifest["source_identity"]["robot"]["path"] == str(selected)
    assert (sources["output"] / "robot/manifest.json").read_bytes() == (selected / "manifest.json").read_bytes()


def test_selected_robot_snapshot_cannot_change_the_model_identity(sources):
    manifest_path = sources["robot"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_checkpoint_sha256"] = "f" * 64
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="exact model bundle"):
        exporter.export_orin(sources["output"], sources["references"], root=sources["root"],
                             client_source=sources["client"], robot_bundle=sources["robot"])
    assert not sources["output"].exists()
