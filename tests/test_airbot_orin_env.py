from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_airbot_orin_env as preparation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def versions():
    return preparation.load_versions(ROOT / "deploy/airbot-paperbag200-da2-100k-20260924/runtime-versions.json")


def host_fixture():
    return {"platform": {"system": "Linux", "machine": "aarch64", "l4t_release": "# R36 (release)",
                         "board_model": "NVIDIA Jetson AGX Orin Developer Kit"},
            "mount": {"is_mountpoint": True, "is_nvme_block_device": True, "is_independent_device": True,
                      "mount": {"options": ["rw"]}, "free_bytes": 10 * 1024**3}}


def base_fixture(site_dir, versions):
    packages = deepcopy(versions["recorded_runtime"]["packages"])
    packages.update(torch="2.5.0a0+nv24.08", torchvision="0.20.0a0+nv")
    return {"status": "probed", "machine": "aarch64", "eligible_cuda_base": True,
            "site_dirs": [str(site_dir)], "packages": packages,
            "torch_identity": {"version": packages["torch"], "module_file": str(site_dir / "torch/__init__.py"),
                               "files_sha256": {str(site_dir / "torch/_C.so"): "base-torch-hash"}},
            "torchvision_identity": {"version": packages["torchvision"],
                                     "module_file": str(site_dir / "torchvision/__init__.py"),
                                     "files_sha256": {str(site_dir / "torchvision/_C.so"): "base-tv-hash"}},
            "protected_distributions": {"torch": packages["torch"], "torchvision": packages["torchvision"]},
            "cuda_runtime": "12.6", "cudnn_version": 90300}


def test_source_runtime_targets_exclude_cuda_and_openpi_installation(versions):
    packages = versions["managed_packages"]
    assert packages["transformers"] == "4.57.6" and packages["numpy"] == "1.26.0"
    assert packages["pillow"] == "11.3.0" and packages["safetensors"] == "0.8.0"
    assert not {"torch", "torchvision", "torchaudio", "openpi-client"} & set(packages)


def test_tool_import_requires_only_stdlib_and_enables_no_hardware():
    code = "import sys; from scripts import prepare_airbot_orin_env; assert not ({'numpy','torch','torchvision','airbot_hardware_py','jax'} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, timeout=10, check=True)


@pytest.mark.parametrize("invalid", ["x86", "no_l4t", "wrong_board", "unmounted", "not_nvme", "readonly", "full_disk"])
def test_create_host_gate_rejects_unsafe_targets(tmp_path, monkeypatch, invalid):
    mount = tmp_path / "nvme"
    mount.mkdir()
    monkeypatch.setattr(preparation, "NVME_ROOT", mount)
    host = host_fixture()
    if invalid == "x86":
        host["platform"]["machine"] = "x86_64"
    elif invalid == "no_l4t":
        host["platform"]["l4t_release"] = None
    elif invalid == "wrong_board":
        host["platform"]["board_model"] = "Other aarch64 system"
    elif invalid == "unmounted":
        host["mount"]["is_mountpoint"] = False
    elif invalid == "not_nvme":
        host["mount"]["is_nvme_block_device"] = False
    elif invalid == "readonly":
        host["mount"]["mount"]["options"] = ["ro"]
    else:
        host["mount"]["free_bytes"] = 1024
    output = mount / "new-venv"
    with pytest.raises(ValueError):
        preparation.validate_create_host(host, output)
    assert not output.exists()


def test_real_nvme_mount_can_share_the_root_nvme_device(tmp_path, monkeypatch):
    mount = tmp_path / "nvme"
    mount.mkdir()
    monkeypatch.setattr(preparation, "NVME_ROOT", mount)
    host = host_fixture()
    host["mount"]["is_independent_device"] = False
    preparation.validate_create_host(host, mount / "new-venv")


def test_create_refuses_existing_outputs_and_symlink_escape(tmp_path, monkeypatch):
    mount = tmp_path / "nvme"
    mount.mkdir()
    monkeypatch.setattr(preparation, "NVME_ROOT", mount)
    existing = mount / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        preparation.validate_create_host(host_fixture(), existing)
    outside = tmp_path / "outside"
    outside.mkdir()
    (mount / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="below the actual"):
        preparation.validate_create_host(host_fixture(), mount / "escape/new-venv")


def test_probe_keeps_the_venv_python_launcher_path(tmp_path, monkeypatch):
    executable = tmp_path / "base-venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    calls = []
    def run(command, timeout):
        calls.append(command)
        return {"returncode": 0, "stdout": 'AIRBOT_ENV_PROBE={"status":"probed","eligible_cuda_base":false}', "stderr": ""}
    monkeypatch.setattr(preparation, "run_readonly", run)
    result = preparation.probe_python(executable)
    assert calls[0][0] == executable
    assert result["requested_python"] == str(executable)
    assert Path(calls[0][0]) != executable.resolve()


def test_pip_targets_are_official_pinned_wheels_without_dependency_resolution(tmp_path, versions):
    base = base_fixture(tmp_path, versions)
    constraints = preparation.constraints_text(base, versions["managed_packages"])
    assert "torch==2.5.0a0+nv24.08\n" in constraints
    assert "torchvision==0.20.0a0+nv\n" in constraints
    command = preparation.install_command("/mnt/nvme/new/bin/python", "/mnt/nvme/new/constraints.txt", versions["managed_packages"])
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--only-binary=:all:" in command and "--no-deps" in command and "--ignore-installed" in command
    targets = [value.split("==", 1)[0] for value in command if "==" in value]
    assert set(targets) == set(preparation.MANAGED_PACKAGES)
    with pytest.raises(ValueError, match="unknown/CUDA"):
        preparation.install_command("python", "constraints", {**versions["managed_packages"], "torch": "2.7.0"})


def test_subprocess_environment_disables_pip_index_injection_and_user_pythonpath(monkeypatch):
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://untrusted.invalid/simple")
    monkeypatch.setenv("PIP_FIND_LINKS", "/untrusted/wheels")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/known/jetson/native")
    monkeypatch.setenv("LD_PRELOAD", "/old/jax-workaround.so")
    monkeypatch.setenv("XLA_FLAGS", "old-jax-options")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    result = preparation.subprocess_environment()
    assert "PIP_EXTRA_INDEX_URL" not in result and "PIP_FIND_LINKS" not in result and "PYTHONPATH" not in result
    assert result["PIP_CONFIG_FILE"] == "/dev/null"
    assert not {"LD_LIBRARY_PATH", "LD_PRELOAD", "XLA_FLAGS", "JAX_PLATFORMS"} & set(result)


@pytest.mark.parametrize("change", ["path", "sha", "cuda", "version"])
def test_protected_identity_detects_shadowed_cuda_packages(tmp_path, versions, change):
    base = base_fixture(tmp_path, versions)
    current = deepcopy(base)
    if change == "path":
        current["torch_identity"]["module_file"] = "/new-venv/torch/__init__.py"
    elif change == "sha":
        current["torchvision_identity"]["files_sha256"][str(tmp_path / "torchvision/_C.so")] = "changed"
    elif change == "cuda":
        current["cuda_runtime"] = "12.8"
    else:
        current["protected_distributions"]["torch"] = "another-version"
    with pytest.raises(ValueError, match="changed or was shadowed"):
        preparation.verify_protected_identity(base, current)


def mocked_create(tmp_path, monkeypatch, versions, *, pip_returncode=0, dependencies_ok=True):
    mount = tmp_path / "nvme"
    mount.mkdir()
    site_dir = tmp_path / "base-venv/lib/python3.10/site-packages"
    site_dir.mkdir(parents=True)
    monkeypatch.setattr(preparation, "NVME_ROOT", mount)
    base = base_fixture(site_dir, versions)
    output = mount / "inference-new"
    args = SimpleNamespace(output=output, base_python=tmp_path / "base-venv/bin/python", openpi_client_root=None,
                           probe_timeout=1, install_timeout=1)
    monkeypatch.setattr(preparation, "inspect_host", lambda *unused: host_fixture())
    monkeypatch.setattr(preparation, "probe_python", lambda *unused: deepcopy(base))
    def run(command, timeout):
        if "venv" in command:
            (output / "bin").mkdir()
            (output / "bin/python").write_text("synthetic interpreter; never executed")
            (output / "lib/python3.10/site-packages").mkdir(parents=True)
            return {"returncode": 0, "stdout": "", "stderr": ""}
        if "AIRBOT_SITE=" in str(command):
            value = {"prefix": str(output), "site": str(output / "lib/python3.10/site-packages")}
            return {"returncode": 0, "stdout": "AIRBOT_SITE=" + json.dumps(value), "stderr": ""}
        if preparation.VERIFY_CODE in command:
            return {"returncode": 0, "stdout": "AIRBOT_ENV_VERIFY=" + json.dumps({
                "passed": dependencies_ok, "dependency_errors": [] if dependencies_ok else [{"requirement": "missing-package"}]}),
                "stderr": ""}
        pytest.fail(f"Unexpected subprocess: {command}")
    monkeypatch.setattr(preparation, "run_readonly", run)
    calls = []
    def pip(command, **kwargs):
        calls.append(command)
        assert "--no-deps" in command
        return SimpleNamespace(returncode=pip_returncode)
    monkeypatch.setattr(preparation.subprocess, "run", pip)
    return args, output, base, calls


def test_create_records_readonly_base_site_inheritance_and_remains_unattested(tmp_path, monkeypatch, versions):
    args, output, base, calls = mocked_create(tmp_path, monkeypatch, versions)
    result = preparation.create_environment(args, versions)
    assert result["status"] == "prepared_pending_attestation"
    assert result["inference_ready"] is False and result["cross_platform_attestation_required"] is True
    assert result["base_environment_unchanged"] and result["protected_cuda_packages_unchanged"]
    assert result["vendored_openpi_client_required"] is True
    inheritance = (output / "lib/python3.10/site-packages/_airbot_verified_base.pth").read_text()
    assert "site.addsitedir" in inheritance and repr(base["site_dirs"][0]) in inheritance
    assert result["cuda_version_differences"]["torch"]["orin_base"] == base["packages"]["torch"]
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        preparation.create_environment(args, versions)


@pytest.mark.parametrize("failure", ["pip", "dependencies"])
def test_failed_creation_is_partial_preserved_and_never_marked_ready(tmp_path, monkeypatch, versions, failure):
    args, output, base, calls = mocked_create(tmp_path, monkeypatch, versions,
                                             pip_returncode=1 if failure == "pip" else 0,
                                             dependencies_ok=failure != "dependencies")
    with pytest.raises(RuntimeError):
        preparation.create_environment(args, versions)
    report = json.loads((output / "setup-state.json").read_text())
    assert report["status"] == "partial" and report["inference_ready"] is False
    assert report["base_environment_unchanged"] is True
    assert report["base_after_probe"]["torch_identity"] == base["torch_identity"]
    assert (output / "install.log").is_file() and len(calls) == 1
    with pytest.raises(FileExistsError):
        preparation.create_environment(args, versions)


def test_create_uses_explicit_local_openpi_snapshot_without_pip_install(tmp_path, monkeypatch, versions):
    args, output, base, calls = mocked_create(tmp_path, monkeypatch, versions)
    vendor = tmp_path / "verified-snapshot"
    (vendor / "openpi_client").mkdir(parents=True)
    (vendor / "openpi_client/__init__.py").write_text("")
    (vendor / "openpi_client/msgpack_numpy.py").write_text("# trusted test snapshot\n")
    args.openpi_client_root = vendor
    report = preparation.create_environment(args, versions)
    assert report["vendored_openpi_client_required"] is False
    assert report["openpi_client_snapshot"]["files_sha256"]["openpi_client/msgpack_numpy.py"]
    assert not any(value.startswith("openpi-client==") for value in calls[0])
