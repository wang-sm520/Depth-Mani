#!/usr/bin/env python3
"""Manual AIRBOT paper-bag deployment; robot motion requires `robot --execute`."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
DEFAULT_PROFILE = ROOT / "configs/airbot_paperbag_deploy.json"


def resolve_path(value, root=ROOT):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inventory(directory, expected_checkpoint=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete package: {directory}")
    if expected_checkpoint and manifest.get("exported_checkpoint_sha256") != expected_checkpoint:
        raise ValueError("Package checkpoint identity differs from the deployment profile")
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = directory / relative
        if (relative.is_absolute() or ".." in relative.parts or item["path"] in seen
                or path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(directory.resolve())):
            raise ValueError(f"Invalid package entry: {item['path']}")
        seen.add(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Package checksum mismatch: {item['path']}")
    if not seen:
        raise ValueError("Package file inventory is empty")
    if expected_checkpoint and ("policy.pt" not in seen or sha256(directory / "policy.pt") != expected_checkpoint):
        raise ValueError("Package does not contain the expected policy.pt")
    return len(seen)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    shared.add_argument("--machine", choices=["workstation", "orin"],
                        default="orin" if platform.machine() == "aarch64" else "workstation")
    shared.add_argument("--openpi-root", type=Path)
    shared.add_argument("--robot-python", type=Path)
    shared.add_argument("--inference-python", type=Path)
    shared.add_argument("--host", default="127.0.0.1")
    shared.add_argument("--port", type=int, default=8026)

    check = commands.add_parser("check", parents=[shared], help="Check software and enumerate devices; no motor enable")
    check.add_argument("--can-interface", help="Also require this confirmed CAN interface to be present and UP")
    commands.add_parser("serve", parents=[shared], help="Start the fixed best76k policy on this GPU workstation")
    probe = commands.add_parser("probe", parents=[shared], help="Send recorded RGB/state to the server, without hardware")
    probe.add_argument("--observation", type=Path)
    probe.add_argument("--report", type=Path)
    cameras = commands.add_parser("cameras", parents=[shared], help="Capture current RGB from both cameras; no robot connection")
    cameras.add_argument("--output-dir", type=Path)
    robot = commands.add_parser("robot", parents=[shared], help="Preview a robot command, or run it with --execute")
    robot.add_argument("--can-interface", required=True, help="Confirmed follower CAN interface on the robot host")
    robot.add_argument("--reset-action", type=float, nargs=7, required=True, metavar="Q")
    robot.add_argument("--max-steps", type=int, default=25)
    robot.add_argument("--chunk-size-execute", type=int, default=4)
    robot.add_argument("--log", type=Path)
    robot.add_argument("--execute", action="store_true", help="Enable/connect hardware; Enter then resets and runs")
    tunnel = commands.add_parser("tunnel", help="Forward the workstation policy to localhost on the robot host")
    tunnel.add_argument("--robot-host", required=True, help="SSH destination, e.g. nvidia@100.68.24.27")
    tunnel.add_argument("--port", type=int, default=8026, help="Workstation policy port")
    tunnel.add_argument("--robot-port", type=int, default=8026)
    return parser


def runtime_settings(args, profile):
    settings = profile["environments"][args.machine].copy()
    for key in ("openpi_root", "robot_python", "inference_python"):
        override = getattr(args, key, None)
        settings[key] = resolve_path(override if override is not None else settings[key])
    settings["native_library_dir"] = resolve_path(settings.get("native_library_dir"))
    return settings


def robot_environment(settings):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("LD_LIBRARY_PATH", None)
    if settings["native_library_dir"] is not None:
        env["LD_LIBRARY_PATH"] = str(settings["native_library_dir"])
    return env


def require_executable(path, label):
    if path is None or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{label} is unavailable: {path}; set the corresponding Python override")


def run_foreground(command, env=None):
    print(shlex.join([str(value) for value in command]), flush=True)
    os.execvpe(str(command[0]), [str(value) for value in command], env or os.environ.copy())


def output_path(kind, suffix):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "runtime/airbot-deploy" / f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}{suffix}"


def client_command(args, settings, profile):
    command = [str(settings["robot_python"]), "-m", "airbot_deploy.client", args.command,
               "--profile", str(args.profile.resolve()), "--openpi-root", str(settings["openpi_root"]),
               "--host", args.host, "--port", str(args.port)]
    if args.command == "probe":
        bundle = resolve_path(profile.get("bundle"))
        sample = ROOT / "sample_observation.npz" if profile.get("client_only") else bundle / "sample_observation.npz"
        command += ["--observation", str(args.observation or sample),
                    "--report", str(args.report or output_path("probe", ".json"))]
    elif args.command == "cameras":
        command += ["--output-dir", str(args.output_dir or output_path("cameras", ""))]
    elif args.command == "robot":
        command += ["--can-interface", args.can_interface, "--reset-action",
                    *[str(value) for value in args.reset_action],
                    "--max-steps", str(args.max_steps), "--chunk-size-execute", str(args.chunk_size_execute),
                    "--log", str(args.log or output_path("robot", ".log"))]
        if args.execute:
            command.append("--execute")
    return command


def check_environment(args, settings, profile):
    result = {"profile": profile["profile_id"], "machine": args.machine,
              "checkpoint_sha256": profile["policy"]["checkpoint_sha256"], "robot_enabled": False}
    if profile.get("client_only"):
        result["verified_client_files"] = verify_inventory(ROOT)
    else:
        result["verified_model_files"] = verify_inventory(resolve_path(profile["bundle"]),
                                                          profile["policy"]["checkpoint_sha256"])
        require_executable(settings["inference_python"], "Inference Python")
    require_executable(settings["robot_python"], "Robot Python")
    hardware_dir = settings["openpi_root"] / "examples/airbot"
    if not hardware_dir.is_dir():
        raise ValueError(f"AIRBOT hardware helper directory is missing: {hardware_dir}")
    code = (
        "import sys,json,importlib,importlib.metadata; sys.path.insert(0,sys.argv[1]); "
        "names=['numpy','websockets.sync.client','openpi_client.msgpack_numpy',"
        "'airbot_hardware_py','camera_sources','play_operator_ah']; "
        "modules={n:str(getattr(importlib.import_module(n),'__file__','built-in')) for n in names}; "
        "versions={n:importlib.metadata.version(n) for n in ['numpy','websockets','airbot-hardware-py']}; "
        "assert versions['airbot-hardware-py']=='0.2.9.2',versions; "
        "print(json.dumps({'python':sys.version,'modules':modules,'versions':versions}))"
    )
    probe = subprocess.run([str(settings["robot_python"]), "-c", code, str(hardware_dir)],
                           env=robot_environment(settings), capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise RuntimeError(f"Robot environment import check failed:\n{probe.stderr[-4000:]}")
    result["robot_environment"] = json.loads(probe.stdout.strip().splitlines()[-1])
    if shutil.which("ip"):
        devices = subprocess.run(["ip", "-details", "-json", "link", "show"], capture_output=True,
                                 text=True, check=True, timeout=5)
        links = json.loads(devices.stdout)
        result["can_interfaces"] = [{"name": item["ifname"], "flags": item.get("flags", [])}
                                    for item in links if item.get("link_type") == "can"]
    else:
        result["can_interfaces"] = []
    if args.can_interface:
        match = [item for item in result["can_interfaces"] if item["name"] == args.can_interface]
        if not match or "UP" not in match[0]["flags"]:
            raise ValueError(f"CAN interface {args.can_interface!r} is absent or DOWN")
    result["status"] = "software_ready"
    result["hardware_capture_tested"] = False
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be in 1..65535")
    if args.command == "tunnel":
        if not 1 <= args.robot_port <= 65535 or args.robot_host.startswith("-"):
            raise ValueError("Invalid SSH destination or forwarded port")
        run_foreground(["ssh", "-N", "-T", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                        "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=yes", "-R",
                        f"127.0.0.1:{args.robot_port}:127.0.0.1:{args.port}", args.robot_host])
        return
    profile = json.loads(args.profile.read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported deployment profile schema")
    settings = runtime_settings(args, profile)
    if args.command == "check":
        check_environment(args, settings, profile)
        return
    if args.command == "serve":
        if profile.get("client_only"):
            raise ValueError("This is the robot client package; start serve in depth-mani on the 5090")
        bundle = resolve_path(profile["bundle"])
        verify_inventory(bundle, profile["policy"]["checkpoint_sha256"])
        require_executable(settings["inference_python"], "Inference Python")
        env = os.environ.copy()
        env["AIRBOT_DEPTH_PYTHON"] = str(settings["inference_python"])
        run_foreground(["bash", str(bundle / "serve.sh"), "--host", args.host, "--port", str(args.port)], env)
        return
    if args.command == "robot":
        frequency = profile.get("robot", {}).get("control_frequency")
        if type(frequency) is not int or frequency != 100:
            raise ValueError("Robot profile must use the verified 100 Hz low-level control frequency")
        if not args.execute:
            print("Preview only: no robot connection or motor enable. Add --execute to connect; Enter then resets/runs.")
            print(f"Low-level control: {frequency} Hz; delayed sends do not catch up missed periods.")
            print(shlex.join(client_command(args, settings, profile)))
            return
    if profile.get("client_only"):
        verify_inventory(ROOT)
    require_executable(settings["robot_python"], "Robot Python")
    run_foreground(client_command(args, settings, profile), robot_environment(settings))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        sys.exit(1)
