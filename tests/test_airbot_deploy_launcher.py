"""Manual deployment must remain inert until explicitly executed."""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/deploy_airbot_paperbag.py"


def _load_launcher(path=LAUNCHER):
    spec = importlib.util.spec_from_file_location(f"deployment_launcher_{id(path)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher():
    return _load_launcher()


def _write_profile(root, **overrides):
    profile = {
        "schema_version": 1, "profile_id": "test-paperbag", "client_only": False,
        "bundle": str(root / "model"), "policy": {"checkpoint_sha256": "0" * 64},
        "robot": {"control_frequency": 100},
        "environments": {
            machine: {"robot_python": "runtime/robot-python", "inference_python": "runtime/policy-python",
                      "openpi_root": "openpi", "native_library_dir": None}
            for machine in ("workstation", "orin")
        },
        **overrides,
    }
    path = root / "configs/profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    return path, profile


def _robot_args(profile):
    return ["robot", "--profile", str(profile), "--machine", "orin", "--can-interface", "can2",
            "--reset-action", "0", "0", "0", "0", "0", "0", "0.07"]


def _forbid_side_effect(*args, **kwargs):
    raise AssertionError("Preview attempted to invoke a deployment operation")


def test_robot_default_does_not_import_hardware_or_run_a_command(launcher, tmp_path, monkeypatch, capsys):
    profile_path, _ = _write_profile(tmp_path)
    real_import = builtins.__import__
    forbidden = {"torch", "jax", "numpy", "airbot_hardware_py", "airbot_deploy", "airbot_depth",
                 "openpi", "openpi_client", "camera_sources", "play_operator_ah"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError(f"Preview imported runtime module {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(launcher, "run_foreground", _forbid_side_effect)
    monkeypatch.setattr(launcher, "require_executable", _forbid_side_effect)
    monkeypatch.setattr(launcher, "verify_inventory", _forbid_side_effect)
    monkeypatch.setattr(launcher.subprocess, "run", _forbid_side_effect)
    launcher.main(_robot_args(profile_path))
    output = capsys.readouterr().out
    assert "Preview only" in output
    assert "Low-level control: 100 Hz" in output
    command = shlex.split(output.strip().splitlines()[-1])
    assert command[1:4] == ["-m", "airbot_deploy.client", "robot"]
    assert "--execute" not in command
    assert command[command.index("--can-interface") + 1] == "can2"


@pytest.mark.parametrize("execute", [False, True])
def test_robot_rejects_unverified_frequency_before_launch(launcher, tmp_path, monkeypatch, execute):
    profile_path, _ = _write_profile(tmp_path, robot={"control_frequency": 250})
    monkeypatch.setattr(launcher, "run_foreground", _forbid_side_effect)
    monkeypatch.setattr(launcher, "verify_inventory", _forbid_side_effect)
    args = _robot_args(profile_path) + (["--execute"] if execute else [])
    with pytest.raises(ValueError, match="100 Hz low-level control frequency"):
        launcher.main(args)


def test_serve_rejects_changed_checkpoint_before_launching(launcher, tmp_path, monkeypatch):
    bundle = tmp_path / "model"
    bundle.mkdir()
    policy = bundle / "policy.pt"
    original = b"original policy bytes"
    policy.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps({
        "status": "complete", "exported_checkpoint_sha256": digest,
        "files": [{"path": "policy.pt", "bytes": len(original), "sha256": digest}],
    }))
    profile_path, _ = _write_profile(tmp_path, policy={"checkpoint_sha256": digest})
    policy.write_bytes(b"different policy data")
    monkeypatch.setattr(launcher, "run_foreground", _forbid_side_effect)
    with pytest.raises(ValueError, match="checksum|checkpoint"):
        launcher.main(["serve", "--profile", str(profile_path), "--machine", "workstation"])


def test_tunnel_listens_on_robot_localhost_and_targets_workstation_localhost(launcher, monkeypatch):
    commands = []
    monkeypatch.setattr(launcher, "run_foreground", lambda command, env=None: commands.append(command))
    launcher.main(["tunnel", "--robot-host", "nvidia@robot.example"])
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == "ssh"
    assert command[-1] == "nvidia@robot.example"
    assert command[command.index("-R") + 1] == "127.0.0.1:8026:127.0.0.1:8026"
    assert "ExitOnForwardFailure=yes" in command
    assert "StrictHostKeyChecking=yes" in command
    assert "-N" in command and "-T" in command


def test_check_without_can_reports_software_only(launcher, tmp_path, monkeypatch, capsys):
    openpi = tmp_path / "openpi"
    (openpi / "examples/airbot").mkdir(parents=True)
    profile_path, _ = _write_profile(tmp_path)
    monkeypatch.setattr(launcher, "verify_inventory", lambda *args, **kwargs: 2)
    monkeypatch.setattr(launcher, "require_executable", lambda *args: None)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/ip" if name == "ip" else None)

    def read_only_probe(command, **kwargs):
        if command[0] == "ip":
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        assert command[1] == "-c"
        assert ".connect(" not in command[2] and ".enable(" not in command[2]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"modules": {}, "versions": {}}), stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", read_only_probe)
    launcher.main(["check", "--profile", str(profile_path), "--openpi-root", str(openpi)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "software_ready"
    assert result["can_interfaces"] == []
    assert result["robot_enabled"] is False
    assert result["hardware_capture_tested"] is False
    with pytest.raises(ValueError, match="absent or DOWN"):
        launcher.main(["check", "--profile", str(profile_path), "--openpi-root", str(openpi),
                       "--can-interface", "can2"])


@pytest.mark.parametrize("inside_scripts", [False, True])
def test_copied_orin_launcher_resolves_paths_from_package_not_cwd(tmp_path, monkeypatch, capsys, inside_scripts):
    package = tmp_path / "copied-client"
    destination = package / "scripts/deploy_airbot_paperbag.py" if inside_scripts else package / "deploy_airbot_paperbag.py"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, destination)
    profile_path, _ = _write_profile(package, client_only=True)
    copied = _load_launcher(destination)
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    copied.main(_robot_args(profile_path))
    command = shlex.split(capsys.readouterr().out.strip().splitlines()[-1])
    assert Path(command[0]) == package / "runtime/robot-python"
    assert Path(command[command.index("--openpi-root") + 1]) == package / "openpi"
    assert Path(command[command.index("--profile") + 1]) == profile_path
    assert Path(command[command.index("--log") + 1]).is_relative_to(package)
