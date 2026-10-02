#!/usr/bin/env python3
"""Recover this deployment's interrupted dependency install from verified local wheels.

The immutable setup driver and original failure evidence are preserved. Only its
ten managed packages, or the separately approved two missing support packages,
may be installed; CUDA and shared environments stay intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import email
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import zipfile


TARGET = Path("/mnt/nvme/pi05/airbot-paperbag200-da2-orin-native-20260924")
MANIFEST_SHA256 = "80e43990f241f5c8be018161e8e4901c866f7700f51bde7f603b6e4830c1d07c"
DRIVER_SHA256 = "ddfd365ccbaf2960296379f21609ed5b8e76d96d2022e8072ae0698733d1b157"
ORIGINAL_STATE_SHA256 = "2f08ee7af62f1326a74ebeb08ba17a579ebd28040b390255cd927cfb0ad46494"
MANAGED = ("numpy", "transformers", "pillow", "huggingface-hub", "h5py", "safetensors",
           "tokenizers", "msgpack", "websockets", "typing-extensions")
STOPPED_PIDS = (35041, 35099)
SUPPORT_PINS = {"regex": "2026.5.9", "hf-xet": "1.5.1"}
SUPPORT_SHA256 = {"regex": "b4bb445ff3f725f59df8f6014edb547ee928ec7023a774f6a39a3f953038cbb2",
                  "hf-xet": "a93df2039190502835b1db8cd7e178b0b7b889fe9ab51299d5ced26e0dd879a4"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expect_hash(path, expected):
    if path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"Original artifact identity changed: {path}")


def load_driver():
    path = TARGET / "prepare_env.py"
    expect_hash(path, DRIVER_SHA256)
    spec = importlib.util.spec_from_file_location("airbot_frozen_env_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.MANAGED_PACKAGES) != MANAGED:
        raise ValueError("Frozen setup driver has another package allowlist")
    return module


def wheel_inventory(directory, expected):
    if directory.is_symlink() or not directory.resolve().is_relative_to(TARGET / "runtime"):
        raise ValueError("Wheel directory must belong to this deployment's runtime")
    records = []
    found = set()
    for path in sorted(directory.glob("*.whl")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Wheels must be ordinary local files")
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"Wheel lacks a unique distribution identity: {path.name}")
            metadata = email.message_from_bytes(archive.read(names[0]))
            name = metadata["Name"].lower().replace("_", "-")
            version = metadata["Version"]
            wheel_name = names[0].removesuffix("METADATA") + "WHEEL"
            tags = email.message_from_bytes(archive.read(wheel_name)).get_all("Tag") or []
        if name not in expected or name in found or version != expected[name]:
            raise ValueError(f"Unrequested, duplicate or incorrectly pinned wheel: {path.name}")
        if not tags or not all(tag.endswith(("_aarch64", "-any")) for tag in tags):
            raise ValueError(f"Wheel is not an ARM/pure-Python artifact: {path.name}")
        found.add(name)
        records.append({"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256(path), "name": name, "version": version, "tags": tags})
    if found != set(expected):
        raise ValueError(f"Offline wheel set is incomplete: {sorted(set(expected) - found)}")
    return records


def preflight():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Recovery is restricted to the actual aarch64 Orin host")
    if TARGET.resolve(strict=True) != TARGET or not TARGET.is_relative_to(Path("/mnt/nvme")):
        raise ValueError("Unexpected deployment path")
    for pid in STOPPED_PIDS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError(f"The previous setup process must be absent before recovery: PID {pid}")
    expect_hash(TARGET / "manifest.json", MANIFEST_SHA256)
    environment = TARGET / "model-env"
    if environment.is_symlink() or not (environment / "pyvenv.cfg").is_file():
        raise ValueError("Recovery requires the existing isolated model venv")
    state_path = environment / "setup-state.json"
    expect_hash(state_path, ORIGINAL_STATE_SHA256)
    state = json.loads(state_path.read_text())
    if (state.get("status") != "partial" or state.get("stage") != "installing_pinned_non_cuda_packages"
            or state.get("output") != str(environment) or state.get("base_environment_unchanged") is not True):
        raise ValueError("Only the recorded interrupted dependency installation may be recovered")
    driver = load_driver()
    versions = driver.load_versions(TARGET / "model/runtime-versions.json")
    if versions["managed_packages"] != state["versions"]["managed_packages"]:
        raise ValueError("Managed package pins differ from the original setup state")
    constraints = environment / "inference-constraints.txt"
    if constraints.read_text() != driver.constraints_text(state["base_probe"], versions["managed_packages"]):
        raise ValueError("Original dependency constraints differ from the recorded base")
    vendor = TARGET / "vendor"
    if state.get("openpi_client_snapshot", {}).get("root") != str(vendor):
        raise ValueError("OpenPI codec snapshot does not belong to the final native package")
    for relative, expected in state["openpi_client_snapshot"]["files_sha256"].items():
        path = vendor / relative
        if not path.resolve().is_relative_to(vendor):
            raise ValueError("Vendored codec path escapes its snapshot")
        expect_hash(path, expected)
    return driver, state, versions, constraints, vendor


def recover(verify_only=False, install_support=False):
    driver, original, versions, constraints, vendor = preflight()
    environment = TARGET / "model-env"
    python = environment / "bin/python"
    base_python = Path(original["base_python"])
    wheels_dir = TARGET / "runtime/inference-wheels"
    packages = versions["managed_packages"]
    wheels = wheel_inventory(wheels_dir, packages)
    support_dir = TARGET / "runtime/support-wheels"
    support = wheel_inventory(support_dir, SUPPORT_PINS) if install_support else []
    if any(item["sha256"] != SUPPORT_SHA256[item["name"]] for item in support):
        raise ValueError("Support wheels differ from the independently transferred official artifacts")
    selected_packages = SUPPORT_PINS if install_support else packages
    selected_directory = support_dir if install_support else wheels_dir
    verification_versions = {**packages, **(SUPPORT_PINS if install_support else {})}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    report_path = TARGET / "runtime" / f"env-recovery-{stamp}.json"
    log_path = report_path.with_suffix(".install.log")
    install_log = environment / "install.log"
    preserved = {str(TARGET / "manifest.json"): MANIFEST_SHA256,
                 str(TARGET / "prepare_env.py"): DRIVER_SHA256,
                 str(environment / "setup-state.json"): ORIGINAL_STATE_SHA256,
                 str(install_log): sha256(install_log), str(constraints): sha256(constraints)}
    report = {
        "schema_version": 1, "operation": "offline_env_recovery", "status": "running",
        "stage": "preflight_passed", "created_at": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path), "environment": str(environment), "python": str(python),
        "base_python": str(base_python), "stopped_setup_pids": list(STOPPED_PIDS),
        "original_artifacts_sha256": preserved, "recovery_driver_sha256": sha256(__file__),
        "managed_packages": packages, "offline_wheels": wheels, "verify_only": verify_only,
        "install_support": install_support, "support_wheels": support,
        "verification_expected_versions": verification_versions,
        "inference_ready": False, "cross_platform_validated": False, "robot_executed": False,
        "original_setup_state_preserved": True, "checks": {}, "commands": [],
    }
    driver.atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "stage": report["stage"]}), flush=True)
    base_before = None
    failure = None
    try:
        base_before = driver.probe_python(base_python, timeout=60)
        driver.verify_protected_identity(original["base_probe"], base_before)
        if base_before["packages"] != original["base_probe"]["packages"]:
            raise ValueError("Base package versions changed since the original setup")
        current = driver.probe_python(python, timeout=60)
        driver.verify_protected_identity(base_before, current)
        if Path(current["prefix"]).resolve() != environment:
            raise ValueError("Selected installer does not belong to this model environment")
        report.update(base_before_probe=base_before, before_probe=current)
        if not verify_only:
            command = [str(python), "-I", "-B", "-m", "pip", "--isolated", "install", "--no-index",
                       "--find-links", str(selected_directory), "--only-binary=:all:", "--no-deps", "--ignore-installed",
                       "--no-cache-dir", "--disable-pip-version-check", "--constraint", str(constraints),
                       *[f"{name}=={version}" for name, version in selected_packages.items()]]
            report["commands"].append(command)
            report.update(stage="installing_offline_managed_packages", install_log=str(log_path))
            driver.atomic_json(report_path, report)
            install_env = driver.subprocess_environment()
            temporary = TARGET / "runtime" / f"env-recovery-{stamp}.tmp"
            temporary.mkdir()
            install_env.update(TMPDIR=str(temporary), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(command, env=install_env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=600, check=False)
            report["install_returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"Offline managed-package install failed: {completed.returncode}")
        report["stage"] = "verifying_original_contract"
        driver.atomic_json(report_path, report)
        after = driver.probe_python(python, timeout=60)
        driver.verify_protected_identity(base_before, after)
        report["after_probe"] = after
        command = [str(python), "-I", "-B", "-c", driver.VERIFY_CODE,
                   json.dumps(verification_versions), str(vendor)]
        result = driver.run_readonly(command, timeout=60)
        report["dependency_verification"] = driver.parse_marked_json(result, "AIRBOT_ENV_VERIFY=")
        report["verification_process"] = {"returncode": result["returncode"], "stderr": result["stderr"]}
        if not report["dependency_verification"].get("passed"):
            raise RuntimeError("Original dependency/import/depth API verification did not pass")
        report["checks"]["original_dependency_contract_passed"] = True
    except Exception as error:
        failure = error
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            base_after = driver.probe_python(base_python, timeout=60)
            driver.verify_protected_identity(original["base_probe"], base_after)
            if base_after["packages"] != original["base_probe"]["packages"]:
                raise ValueError("Read-only base package versions changed")
            if base_before is not None:
                driver.verify_protected_identity(base_before, base_after)
                if base_before["packages"] != base_after["packages"]:
                    raise ValueError("Read-only base packages changed during recovery")
            for path, expected in preserved.items():
                expect_hash(Path(path), expected)
            if wheel_inventory(wheels_dir, packages) != wheels:
                raise ValueError("Offline wheels changed during recovery")
            if support and wheel_inventory(support_dir, SUPPORT_PINS) != support:
                raise ValueError("Support wheels changed during recovery")
            report["base_after_probe"] = base_after
            report["checks"].update(base_environment_unchanged=True, base_cuda_hashes_unchanged=True,
                                     original_artifacts_unchanged=True, offline_wheels_unchanged=True)
        except Exception as error:
            report["preservation_error"] = {"type": type(error).__name__, "message": str(error)}
            failure = failure or error
        report.update(status="passed" if failure is None else "failed",
                      stage="verified_pending_native_attestation" if failure is None else "stopped",
                      completed_at=datetime.now(timezone.utc).isoformat())
        driver.atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "status": report["status"], "checks": report["checks"],
                      "dependency_verification": report.get("dependency_verification"),
                      "error": report.get("error"), "preservation_error": report.get("preservation_error"),
                      "inference_ready": False, "robot_executed": False}), flush=True)
    return report["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Repeat the original checks without installing anything")
    mode.add_argument("--install-support", action="store_true", help="Install only the two approved, hash-pinned missing dependencies")
    args = parser.parse_args()
    lock_path = TARGET / "runtime/env-recovery.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another environment recovery is already running") from error
        return 0 if recover(args.verify_only, args.install_support) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Offline environment recovery failed: {error}", file=sys.stderr)
        raise SystemExit(1)
