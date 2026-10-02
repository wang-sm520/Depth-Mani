"""Gate the local Orin launcher without model imports, setup, or robot motion."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from scripts import prepare_airbot_orin_env as environment_preparation


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "scripts/deploy_airbot_orin.py"


def load_launcher(path):
    spec = importlib.util.spec_from_file_location(f"orin_launcher_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "copied-orin-package"
    root.mkdir()
    shutil.copyfile(LAUNCHER, root / "deploy.py")
    launcher = load_launcher(root / "deploy.py")
    payloads = {"prepare_env.py": "# inert test helper\n",
                "airbot_orin/__init__.py": "# no model imports\n",
                "airbot_orin/runtime.py": "raise RuntimeError('test must not execute model code')\n",
                "robot/deploy.py": "raise RuntimeError('test must not execute a robot command')\n",
                "vendor/openpi_client/__init__.py": '__version__ = "0.1.0"\n',
                "vendor/openpi_client/msgpack_numpy.py": "# no protocol import during launcher tests\n",
                "model/policy.pt": "opaque synthetic policy bytes",
                "model/sample_observation.npz": "opaque observation bytes"}
    for relative, content in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    checkpoint_sha = launcher.sha256(root / "model/policy.pt")
    write_json(root / "model/manifest.json", {"status": "complete", "exported_checkpoint_sha256": checkpoint_sha})
    write_json(root / "references/reference.json", {"schema_version": 1, "status": "complete",
                                                     "checkpoint_sha256": checkpoint_sha})
    shutil.copyfile(PROJECT / "deploy/airbot-paperbag200-da2-100k-20260924/runtime-versions.json",
                    root / "model/runtime-versions.json")
    profile = json.loads((PROJECT / "configs/airbot_paperbag_deploy.json").read_text())
    profile["policy"]["checkpoint_sha256"] = checkpoint_sha
    profile["client_only"] = True
    write_json(root / "robot/configs/airbot_paperbag_deploy.json", profile)
    manifest = {"schema_version": 1, "status": "complete", "bundle_type": "airbot_orin_local",
                "policy_checkpoint_sha256": checkpoint_sha,
                "files": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                           "sha256": launcher.sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]}
    write_json(root / "manifest.json", manifest)
    python = tmp_path / "model-venv/bin/python"
    robot_python = tmp_path / "existing-robot/robot-venv/bin/python"
    for executable in (python, robot_python):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)
    library = robot_python.parent.parent.parent / "native/lib"
    library.mkdir(parents=True)
    attestation = {"schema_version": 1, "status": "passed", "checkpoint_sha256": checkpoint_sha,
                   "package_manifest_sha256": launcher.sha256(root / "manifest.json"),
                   "reference_sha256": launcher.sha256(root / "references/reference.json"),
                   "runtime_source_sha256": launcher.sha256(root / "airbot_orin/runtime.py"),
                   "runtime_depth_provenance": {"libraries": {"torch": "native-Orin"}},
                   "environment": {"device": "cuda", "threads": 4, "python_launcher": str(python),
                                   "python_executable": str(python.resolve()),
                                   "python_prefix": str(python.parent.parent), "machine": "aarch64"},
                   "orin_agx_runtime": True, "robot_executed": False}
    report_path = root / "runtime/attestation.json"
    return SimpleNamespace(root=root, launcher=launcher, python=python, robot_python=robot_python,
                           library=library, manifest=manifest, profile=profile, attestation=attestation,
                           report=report_path, checkpoint_sha=checkpoint_sha)


def activate(package):
    write_json(package.report, package.attestation)
    args = SimpleNamespace(robot_python=package.robot_python, native_library_dir=package.library)
    package.launcher.activate(args, package.python, package.report)
    return json.loads((package.root / "runtime/active.json").read_text())


def robot_argv(execute=False):
    result = ["robot", "--can-interface", "can2", "--reset-action", "0", "-0.5", "0.4", "0", "0", "0", "0.02"]
    return [*result, "--execute"] if execute else result


@pytest.mark.parametrize("command", [["serve"], robot_argv(), robot_argv(execute=True)])
def test_no_attestation_blocks_serve_and_robot_before_subprocess(package, monkeypatch, command):
    calls = []
    monkeypatch.setattr(package.launcher, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="validated local environment|validat"):
        package.launcher.main(command)
    assert calls == []


@pytest.mark.parametrize("field", ["status", "checkpoint_sha256", "package_manifest_sha256", "reference_sha256", "robot_executed"])
def test_activation_requires_a_passed_same_checkpoint_package_record(package, field):
    package.attestation[field] = True if field == "robot_executed" else "wrong"
    write_json(package.report, package.attestation)
    args = SimpleNamespace(robot_python=package.robot_python, native_library_dir=package.library)
    with pytest.raises(ValueError):
        package.launcher.activate(args, package.python, package.report)
    assert not (package.root / "runtime/active.json").exists()


def test_setup_matches_environment_tool_cli_and_does_not_touch_existing_environment(package, monkeypatch):
    calls = []
    monkeypatch.setattr(package.launcher, "run", lambda command, **kwargs: calls.append((list(map(str, command)), kwargs)))
    base = package.python
    original = base.read_bytes()
    new_env = package.root / "model-env"
    package.launcher.main(["setup", "--base-python", str(base), "--env-dir", str(new_env)])
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert Path(command[1]) == package.root / "prepare_env.py"
    parsed = environment_preparation.parse_args(command[2:])
    assert parsed.command == "create" and parsed.base_python == base and parsed.output == new_env
    assert parsed.runtime_versions == package.root / "model/runtime-versions.json"
    assert parsed.openpi_client_root == package.root / "vendor"
    assert base.read_bytes() == original and not new_env.exists()
    assert not kwargs.get("replace", False)


def test_robot_defaults_to_preview_and_execute_is_only_explicit(package, monkeypatch):
    state = activate(package)
    calls = []
    monkeypatch.setattr(package.launcher, "run", lambda command, **kwargs: calls.append((list(map(str, command)), kwargs)))
    package.launcher.main(robot_argv())
    command, options = calls[-1]
    assert "--execute" not in command
    assert command[2] == "robot" and Path(command[1]) == package.root / "robot/deploy.py"
    assert command[command.index("--profile") + 1] == state["profile"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--chunk-size-execute") + 1] == "4"
    package.launcher.main(robot_argv(execute=True))
    assert calls[-1][0].count("--execute") == 1


@pytest.mark.parametrize("tamper", ["profile_bytes", "attestation_bytes", "active_attestation_sha", "active_profile_sha",
                                    "active_python", "active_device", "active_threads"])
def test_changed_active_or_profile_identity_stops_before_launch(package, monkeypatch, tamper):
    state = activate(package)
    if tamper == "profile_bytes":
        Path(state["profile"]).write_text("{}")
    elif tamper == "attestation_bytes":
        package.report.write_text("{}")
    else:
        name = {"active_attestation_sha": "attestation_sha256", "active_profile_sha": "profile_sha256",
                "active_python": "python", "active_device": "device", "active_threads": "threads"}[tamper]
        state[name] = {"python": str(package.robot_python), "device": "cpu", "threads": 8}.get(name, "wrong")
        write_json(package.root / "runtime/active.json", state)
    calls = []
    monkeypatch.setattr(package.launcher, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        package.launcher.main(["serve"])
    assert calls == []


@pytest.mark.parametrize("tamper", ["listed_source", "unlisted_source", "symlink", "duplicate_entry", "unsafe_entry"])
def test_local_package_code_integrity_is_checked_before_any_operation(package, monkeypatch, tamper):
    if tamper == "listed_source":
        (package.root / "airbot_orin/runtime.py").write_text("print('changed code')")
    elif tamper == "unlisted_source":
        (package.root / "torch.py").write_text("raise RuntimeError('unlisted import shadow')")
    elif tamper == "symlink":
        source = package.root / "airbot_orin/runtime.py"
        outside = package.root.parent / "outside-runtime.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    else:
        manifest = deepcopy(package.manifest)
        if tamper == "duplicate_entry":
            manifest["files"].append(deepcopy(manifest["files"][0]))
        else:
            manifest["files"][0]["path"] = "../outside.py"
        write_json(package.root / "manifest.json", manifest)
    calls = []
    monkeypatch.setattr(package.launcher, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        package.launcher.main(["inspect"])
    assert calls == []


def test_package_copied_to_another_path_does_not_depend_on_working_directory(package, tmp_path, monkeypatch):
    destination = tmp_path / "new-location/portable-package"
    shutil.copytree(package.root, destination)
    launcher = load_launcher(destination / "deploy.py")
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = []
    monkeypatch.setattr(launcher, "run", lambda command, **kwargs: calls.append(list(map(str, command))))
    launcher.main(["inspect", "--python", str(package.python)])
    command = calls[0]
    assert Path(command[1]) == destination / "prepare_env.py"
    assert Path(command[command.index("--runtime-versions") + 1]) == destination / "model/runtime-versions.json"
    assert Path(command[command.index("--report") + 1]).is_relative_to(destination / "runtime")
    assert command[command.index("--python") + 1] == str(package.python)
    help_result = subprocess.run([sys.executable, str(destination / "deploy.py"), "--help"], cwd=elsewhere,
                                 capture_output=True, text=True, check=True, timeout=10)
    assert "setup" in help_result.stdout and "serve" in help_result.stdout


def test_runtime_environment_clears_old_jax_and_cuda_loader_overrides(package, monkeypatch):
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "XLA_FLAGS", "JAX_PLATFORMS", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(key, "old-runtime-override")
    environment = package.launcher.runtime_environment()
    assert not {"LD_PRELOAD", "LD_LIBRARY_PATH", "XLA_FLAGS", "JAX_PLATFORMS", "TRANSFORMERS_CACHE"} & set(environment)
    assert environment["PYTHONPATH"].split(":") == [str(package.root), str(package.root / "model"), str(package.root / "vendor")]
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["TRANSFORMERS_OFFLINE"] == "1"
