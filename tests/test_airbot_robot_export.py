import json
from pathlib import Path

import pytest

from scripts.export_airbot_robot import (
    HARDWARE_FILES, PROFILE_PATH, export_robot, publish_new, sha256, verified_inventory,
)


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


@pytest.fixture
def sources(tmp_path):
    root, openpi = tmp_path / "project", tmp_path / "openpi"
    root.mkdir()
    for name in HARDWARE_FILES:
        write(openpi / "examples/airbot" / name, f"# frozen fixture {name}\n")
    write(openpi / ".venv/should_not_copy", "private environment")
    write(openpi / "examples/airbot/other.py", "# unrelated\n")
    write(root / "scripts/deploy_airbot_paperbag.py", "# launcher fixture; never executed\n")
    write(root / "airbot_deploy/__init__.py", "# package\n")
    write(root / "airbot_deploy/client.py", "raise RuntimeError('must not import hardware while exporting')\n")
    write(root / "airbot_deploy/notes.txt", "not runtime source")
    write(root / "docs/airbot-paperbag-real-robot.md", "# Robot client instructions\n")
    bundle = root / "deploy/model"
    write(bundle / "policy.pt", "model fixture; must not be copied")
    write(bundle / "sample_observation.npz", "opaque audited sample fixture")
    checkpoint_sha = sha256(bundle / "policy.pt")
    validation = {"status": "passed", "exported_checkpoint_sha256": checkpoint_sha,
                  "source_checkpoint_sha256": "a" * 64, "action_max_absolute_difference": 0.0,
                  "bundle_actions": [[0.1] * 7 for _ in range(8)], "robot_executed": False}
    write(bundle / "bundle-validation.json", json.dumps(validation))
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(bundle.iterdir())]
    manifest = {"schema_version": 1, "status": "complete", "files": inventory,
                "exported_checkpoint_sha256": checkpoint_sha, "source_checkpoint_sha256": "a" * 64,
                "scope": {"formal_training_run": True, "run_completed_steps": 100000, "checkpoint_training_step": 76000}}
    write(bundle / "manifest.json", json.dumps(manifest))
    profile = {"schema_version": 1, "profile_id": "fixture", "bundle": "deploy/model",
               "policy": {"checkpoint_sha256": checkpoint_sha, "horizon": 8, "action_dim": 7},
               "robot": {"control_frequency": 100},
               "environments": {"workstation": {"openpi_root": str(openpi), "robot_python": "/existing/python"},
                                "orin": {"openpi_root": "/old/orin/openpi", "robot_python": "/robot/python"}}}
    write(root / PROFILE_PATH, json.dumps(profile))
    return root, openpi, bundle, tmp_path / "client-bundle"


def test_export_is_complete_client_only_and_preserves_copied_bytes(sources):
    root, openpi, bundle, output = sources
    original_profile = (root / PROFILE_PATH).read_bytes()
    original_model = (bundle / "manifest.json").read_bytes()
    manifest = export_robot(output, root=root)
    assert manifest["client_only"] is True and manifest["robot_executed"] is False
    assert manifest["policy_checkpoint_sha256"] == sha256(bundle / "policy.pt")
    expected = {"deploy.py", "README.md", "sample_observation.npz", "reference_actions.json", "SHA256SUMS", "manifest.json",
                PROFILE_PATH, "airbot_deploy/__init__.py", "airbot_deploy/client.py",
                *[f"vendor/openpi/examples/airbot/{name}" for name in HARDWARE_FILES]}
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == expected
    verified_inventory(output, manifest)
    profile = json.loads((output / PROFILE_PATH).read_text())
    assert profile["bundle"] is None and profile["client_only"] is True
    assert profile["robot"]["control_frequency"] == 100
    assert all(environment["openpi_root"] == "vendor/openpi" for environment in profile["environments"].values())
    assert profile["environments"]["orin"]["robot_python"] == "/robot/python"
    reference = json.loads((output / "reference_actions.json").read_text())
    assert reference["actions"] == [[0.1] * 7 for _ in range(8)]
    assert reference["sample_observation_sha256"] == sha256(output / "sample_observation.npz")
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert digest == sha256(output / relative)
    assert not any(path.is_symlink() for path in output.rglob("*"))
    assert (root / PROFILE_PATH).read_bytes() == original_profile
    assert (bundle / "manifest.json").read_bytes() == original_model
    assert not list(output.parent.glob(f".{output.name}.stage-*"))


def test_export_refuses_existing_destination_without_touching_it(sources):
    root, _, _, output = sources
    output.mkdir()
    (output / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        export_robot(output, root=root)
    assert (output / "user-file").read_text() == "keep"


def test_atomic_publish_does_not_replace_racing_empty_directory(tmp_path):
    stage, destination = tmp_path / "stage", tmp_path / "destination"
    stage.mkdir()
    (stage / "file").write_text("new")
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish_new(stage, destination)
    assert not list(destination.iterdir())
    assert (stage / "file").read_text() == "new"


@pytest.mark.parametrize("failure", ["tampered_sample", "wrong_policy", "model_bundle_output", "missing_source"])
def test_failed_preflight_never_publishes_partial_package(sources, failure):
    root, openpi, bundle, output = sources
    if failure == "tampered_sample":
        (bundle / "sample_observation.npz").write_text("changed since model audit")
    elif failure == "wrong_policy":
        profile = json.loads((root / PROFILE_PATH).read_text())
        profile["policy"]["checkpoint_sha256"] = "b" * 64
        (root / PROFILE_PATH).write_text(json.dumps(profile))
    elif failure == "model_bundle_output":
        output = bundle / "must-not-write"
    else:
        (openpi / "examples/airbot/rtsp_camera.py").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        export_robot(output, root=root)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.stage-*"))


@pytest.mark.parametrize("escape", ["bundle_dotdot", "inventory_dotdot", "hardware_symlink", "client_symlink"])
def test_source_and_manifest_paths_cannot_escape_declared_roots(sources, escape):
    root, openpi, bundle, output = sources
    outside = root.parent / "outside.py"
    outside.write_text("# outside source boundary\n")
    if escape == "bundle_dotdot":
        profile = json.loads((root / PROFILE_PATH).read_text())
        profile["bundle"] = "../outside"
        (root / PROFILE_PATH).write_text(json.dumps(profile))
    elif escape == "inventory_dotdot":
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["files"][0]["path"] = "../../outside.py"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = (openpi / "examples/airbot/airbot_arm.py" if escape == "hardware_symlink" else root / "airbot_deploy/client.py")
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        export_robot(output, root=root)
    assert not output.exists()
