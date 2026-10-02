#!/usr/bin/env python3
"""Prepare and manually run a complete local AIRBOT depth policy on Orin NVMe."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_package():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("bundle_type") != "airbot_orin_local"
            or not manifest.get("files")):
        raise ValueError("This command requires the assembled Orin package")
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or item["path"] in seen
                or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve())
                or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]):
            raise ValueError(f"Package integrity failed: {item['path']}")
        seen.add(item["path"])
    required = {"deploy.py", "prepare_env.py", "airbot_orin/runtime.py", "model/policy.pt",
                "references/reference.json", "robot/deploy.py", "vendor/openpi_client/msgpack_numpy.py"}
    if not required <= seen:
        raise ValueError("Orin package is missing required code, policy or reference entries")
    code_paths = list(ROOT.glob("*"))
    for directory in ("airbot_orin", "model", "robot", "vendor"):
        code_paths.extend((ROOT / directory).rglob("*"))
    for path in code_paths:
        if path.is_dir() and (path / "__init__.py").is_file():
            code_paths.append(path / "__init__.py")
        if path.is_file() and (path.suffix in {".py", ".pyc", ".so", ".pyd"} or ".so." in path.name):
            if str(path.relative_to(ROOT)) not in seen:
                raise ValueError(f"Unlisted code could shadow verified imports: {path.relative_to(ROOT)}")
    return manifest


def runtime_environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(str(ROOT / name) for name in ("", "model", "vendor")),
               HF_HUB_CACHE=str(ROOT / "model/hf_hub"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "TRANSFORMERS_CACHE", "XLA_FLAGS", "JAX_PLATFORMS"):
        env.pop(key, None)
    return env


def executable(path):
    path = Path(path).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python executable is unavailable: {path}")
    return path


def run(command, *, replace=False, env=None):
    command = [str(value) for value in command]
    print(shlex.join(command), flush=True)
    if replace:
        os.chdir(ROOT)
        os.execvpe(command[0], command, env or os.environ.copy())
    return subprocess.run(command, env=env, cwd=ROOT, check=True)


def new_report(kind):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "runtime" / f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def hardware_runtime(args):
    candidates = [Path(value) for value in (
        "/mnt/nvme/pi05/runtime/official-20260910/robot-venv/bin/python",
        "/mnt/nvme/pi05/official-20260910/robot-venv/bin/python",
    )]
    python = executable(args.robot_python) if args.robot_python else next((path for path in candidates if path.is_file()), None)
    if python is None:
        raise ValueError("Existing Orin robot Python was not found; specify --robot-python")
    library = Path(args.native_library_dir).resolve() if args.native_library_dir else python.parent.parent.parent / "native/lib"
    if not library.is_dir():
        raise ValueError(f"Existing robot native library directory is missing: {library}; specify --native-library-dir")
    return python, library


def activate(args, python, report):
    attestation = json.loads(Path(report).read_text())
    if attestation.get("status") != "passed":
        raise ValueError("Native runtime validation did not pass")
    if (attestation.get("package_manifest_sha256") != sha256(ROOT / "manifest.json")
            or attestation.get("reference_sha256") != sha256(ROOT / "references/reference.json")
            or attestation.get("environment", {}).get("python_launcher") != str(python)
            or attestation.get("robot_executed") is not False):
        raise ValueError("Validation record is not bound to this exact package and fixtures")
    robot_python, library = hardware_runtime(args)
    profile = json.loads((ROOT / "robot/configs/airbot_paperbag_deploy.json").read_text())
    if attestation.get("checkpoint_sha256") != profile["policy"]["checkpoint_sha256"]:
        raise ValueError("Native validation refers to a different policy checkpoint")
    profile["policy"].update(runtime_attestation_sha256=sha256(report),
                              runtime_depth_provenance=deepcopy(attestation["runtime_depth_provenance"]))
    for settings in profile["environments"].values():
        settings.update(robot_python=str(robot_python), openpi_root=str(ROOT / "robot/vendor/openpi"),
                        native_library_dir=str(library))
    # Source references use stricter original-GPU actions. Target probe records
    # transport/shape separately; target numeric acceptance lives in attestation.
    profile_path = new_report("robot-profile")
    atomic_json(profile_path, profile)
    state = {"schema_version": 1, "status": "validated", "python": str(python),
             "device": attestation["environment"]["device"], "threads": attestation["environment"]["threads"],
             "attestation": str(Path(report).resolve()), "attestation_sha256": sha256(report),
             "profile": str(profile_path), "profile_sha256": sha256(profile_path),
             "package_manifest_sha256": sha256(ROOT / "manifest.json"),
             "orin_agx_runtime": attestation.get("orin_agx_runtime", False), "robot_executed": False}
    atomic_json(ROOT / "runtime/active.json", state)
    print(json.dumps({"status": "validated", "attestation": str(report),
                      "orin_agx_runtime": state["orin_agx_runtime"], "robot_executed": False}))


def active():
    path = ROOT / "runtime/active.json"
    if not path.is_file():
        raise ValueError("No validated local environment. Run inspect, setup and validate on this Orin first")
    state = json.loads(path.read_text())
    if (state.get("status") != "validated" or state.get("package_manifest_sha256") != sha256(ROOT / "manifest.json")
            or sha256(state["attestation"]) != state.get("attestation_sha256")
            or sha256(state["profile"]) != state.get("profile_sha256")):
        raise ValueError("Local validation/profile changed; repeat native validation")
    attestation = json.loads(Path(state["attestation"]).read_text())
    runtime = attestation.get("environment", {})
    if (attestation.get("status") != "passed" or state.get("python") != runtime.get("python_launcher")
            or state.get("device") != runtime.get("device") or state.get("threads") != runtime.get("threads")):
        raise ValueError("Selected Python/device/threads differ from the validated native runtime")
    return state


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read-only native GPU/Python/NVMe inventory")
    inspect.add_argument("--python", action="append", default=[])
    inspect.add_argument("--report", type=Path)
    setup = commands.add_parser("setup", help="Create a new NVMe model environment using an existing CUDA Torch")
    setup.add_argument("--base-python", required=True, type=Path)
    setup.add_argument("--env-dir", type=Path, default=ROOT / "model-env")
    validation = commands.add_parser("validate", help="Compare all twelve fixed RGB/depth/action fixtures")
    validation.add_argument("--python", type=Path, default=ROOT / "model-env/bin/python")
    validation.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    validation.add_argument("--threads", type=int, default=4)
    validation.add_argument("--report", type=Path)
    validation.add_argument("--no-activate", action="store_true", help="Save comparison only; do not create a robot profile")
    validation.add_argument("--robot-python", type=Path)
    validation.add_argument("--native-library-dir", type=Path)
    for name in ("serve", "check", "probe", "cameras", "robot"):
        sub = commands.add_parser(name)
        sub.add_argument("--port", type=int, default=8026)
        if name == "robot":
            sub.add_argument("--can-interface", required=True)
            sub.add_argument("--reset-action", type=float, nargs=7, required=True)
            sub.add_argument("--max-steps", type=int, default=25)
            sub.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "port") and not 1 <= args.port <= 65535:
        raise ValueError("Port must be in 1..65535")
    verify_package()
    versions = ROOT / "model/runtime-versions.json"
    if args.command == "inspect":
        command = [sys.executable, ROOT / "prepare_env.py", "inspect", "--runtime-versions", versions,
                   "--report", args.report.absolute() if args.report else new_report("inspection")]
        for python in args.python:
            command += ["--python", Path(python).expanduser().absolute()]
        run(command)
        return
    if args.command == "setup":
        run([sys.executable, ROOT / "prepare_env.py", "create", "--runtime-versions", versions,
             "--base-python", executable(args.base_python), "--output", args.env_dir.expanduser().absolute(),
             "--openpi-client-root", ROOT / "vendor"])
        print("Environment preparation finished. Next run validate with this environment's Python.")
        return
    if args.command == "validate":
        python = executable(args.python)
        report = args.report.absolute() if args.report else new_report("attestation")
        run([python, "-m", "airbot_orin.runtime", "validate", "--root", ROOT,
             "--device", args.device, "--threads", args.threads, "--output", report], env=runtime_environment())
        if not args.no_activate:
            activate(args, python, report)
        return
    state = active()
    if args.command == "serve":
        run([executable(state["python"]), "-m", "airbot_orin.runtime", "serve", "--root", ROOT,
             "--attestation", state["attestation"], "--device", state["device"],
             "--threads", state["threads"], "--port", args.port], replace=True, env=runtime_environment())
        return
    command = [sys.executable, ROOT / "robot/deploy.py", args.command, "--profile", state["profile"],
               "--host", "127.0.0.1", "--port", args.port]
    if args.command == "probe":
        # This duplicate sample has no neighbouring original-GPU reference file.
        # The full target-specific comparisons were already bound into active.json.
        command += ["--observation", ROOT / "model/sample_observation.npz"]
    if args.command == "robot":
        command += ["--can-interface", args.can_interface, "--reset-action", *args.reset_action,
                    "--max-steps", args.max_steps, "--chunk-size-execute", "4"]
        if args.execute:
            command.append("--execute")
    run(command, replace=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Orin deployment stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
