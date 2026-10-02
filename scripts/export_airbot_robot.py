"""Export the paper-bag robot client with hardware-source snapshots and no weights.

Exports are assembled and verified in a sibling temporary directory, then
published atomically without replacing any existing path. No source module is
imported, no service starts, and no robot or remote host is contacted.
"""

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = "configs/airbot_paperbag_deploy.json"
HARDWARE_FILES = ("airbot_arm.py", "play_operator_ah.py", "robot_config.py", "camera_sources.py", "rtsp_camera.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or any(char in value for char in "\r\n\0"):
        raise ValueError("Bundle paths must be plain relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Bundle path escapes its root: {value!r}")
    return str(path)


def inside(root, relative, *, directory=False):
    root = Path(root).resolve(strict=True)
    path = (root / relative_path(relative)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes its declared root: {relative}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Source is not the expected {'directory' if directory else 'file'}: {path}")
    return path


def verified_inventory(root, manifest):
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Source model bundle must have a complete schema-1 manifest")
    result = {}
    for item in manifest.get("files", []):
        relative = relative_path(item["path"])
        if relative in result:
            raise ValueError(f"Duplicate bundle inventory entry: {relative}")
        path = inside(root, relative)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Bundle integrity check failed: {relative}")
        result[relative] = item
    if not result:
        raise ValueError("Bundle inventory is empty")
    return result


def _write_text(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _copy(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256(destination) != expected:
        raise ValueError(f"Source changed while copying: {source}")


def publish_new(stage, destination):
    """Linux renameat2 prevents even a racing empty destination being replaced."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace export requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def export_robot(output, openpi_root=None, *, root=ROOT, profile_path=PROFILE_PATH, readme_path=None):
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite robot-client export: {output}")
    root = Path(root).resolve(strict=True)
    profile_candidate = Path(profile_path).expanduser()
    if profile_candidate.is_absolute():
        profile_relative = None
        profile_source = profile_candidate.resolve(strict=True)
    else:
        profile_relative = relative_path(profile_path)
        profile_source = inside(root, profile_relative)
    profile_hash = sha256(profile_source)
    profile = json.loads(profile_source.read_text())
    if profile.get("schema_version") != 1 or profile.get("client_only") is True:
        raise ValueError("Export requires the original deployment profile with its model bundle")
    bundle_value = profile.get("bundle")
    if not isinstance(bundle_value, str) or not bundle_value:
        raise ValueError("Deployment profile must identify its model bundle")
    bundle_candidate = Path(bundle_value).expanduser()
    bundle = (bundle_candidate.resolve(strict=True) if bundle_candidate.is_absolute()
              else inside(root, bundle_value, directory=True))
    if not bundle.is_dir():
        raise ValueError(f"Model bundle is not a directory: {bundle}")
    output = output.parent.resolve() / output.name
    if output.is_relative_to(bundle):
        raise ValueError("Robot-client output must be separate from the immutable model bundle")
    model_manifest_path = inside(bundle, "manifest.json")
    model_manifest_hash = sha256(model_manifest_path)
    model_manifest = json.loads(model_manifest_path.read_text())
    inventory = verified_inventory(bundle, model_manifest)
    if not {"sample_observation.npz", "bundle-validation.json", "policy.pt"} <= set(inventory):
        raise ValueError("Model bundle lacks its audited sample, policy or validation record")
    scope = model_manifest.get("scope", {})
    if profile_relative == PROFILE_PATH:
        if (scope.get("formal_training_run") is not True
                or scope.get("run_completed_steps") != 100000):
            raise ValueError("This robot-client package requires the formal 100000-step model bundle")
    elif (type(scope.get("run_completed_steps")) is not int or scope.get("run_completed_steps") < 1
          or type(scope.get("checkpoint_training_step")) is not int
          or not 1 <= scope.get("checkpoint_training_step") <= scope.get("run_completed_steps")):
        raise ValueError("Custom robot profile requires a completed model-bundle training scope")
    checkpoint_hash = model_manifest["exported_checkpoint_sha256"]
    if checkpoint_hash != profile["policy"]["checkpoint_sha256"] or inventory["policy.pt"]["sha256"] != checkpoint_hash:
        raise ValueError("Deployment profile and model bundle specify different policy checkpoints")
    validation_path = inside(bundle, "bundle-validation.json")
    validation = json.loads(validation_path.read_text())
    difference = validation.get("action_max_absolute_difference", float("inf"))
    if (validation.get("status") != "passed" or validation.get("exported_checkpoint_sha256") != checkpoint_hash
            or validation.get("source_checkpoint_sha256") != model_manifest["source_checkpoint_sha256"]
            or validation.get("robot_executed") is not False or not math.isfinite(difference) or difference > 1e-6):
        raise ValueError("Model bundle lacks a consistent passed offline inference comparison")
    actions = validation["bundle_actions"]
    horizon, dimension = profile["policy"]["horizon"], profile["policy"]["action_dim"]
    if (not isinstance(actions, list) or len(actions) != horizon
            or any(not isinstance(row, list) or len(row) != dimension for row in actions)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for row in actions for value in row)):
        raise ValueError("Reference actions must be a finite array matching the policy horizon/dimensions")
    openpi_root = Path(openpi_root or profile["environments"]["workstation"]["openpi_root"]).expanduser().resolve(strict=True)
    client_dir = inside(root, "airbot_deploy", directory=True)
    modules = sorted(client_dir.glob("*.py"))
    if not {"__init__.py", "client.py"} <= {path.name for path in modules}:
        raise FileNotFoundError("airbot_deploy/__init__.py and client.py are required before export")
    if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", path.name) is None for path in modules):
        raise ValueError("Client Python filenames must be valid plain module names")
    readme_source = (Path(readme_path).expanduser().resolve(strict=True) if readme_path is not None
                     else inside(root, "docs/airbot-paperbag-real-robot.md"))
    sources = [(inside(root, "scripts/deploy_airbot_paperbag.py"), "deploy.py"),
               (readme_source, "README.md"),
               (inside(bundle, "sample_observation.npz"), "sample_observation.npz")]
    sources.extend((inside(root, f"airbot_deploy/{path.name}"), f"airbot_deploy/{path.name}") for path in modules)
    hardware = []
    for name in HARDWARE_FILES:
        source = inside(openpi_root, f"examples/airbot/{name}")
        destination = f"vendor/openpi/examples/airbot/{name}"
        sources.append((source, destination))
        hardware.append({"source_path": str(source), "bundle_path": destination})
    copies = [(source, destination, sha256(source)) for source, destination in sources]
    copy_hashes = {destination: digest for _, destination, digest in copies}
    for item in hardware:
        item["sha256"] = copy_hashes[item["bundle_path"]]
    client_profile = deepcopy(profile)
    client_profile.update(client_only=True, bundle=None)
    for environment in client_profile["environments"].values():
        environment["openpi_root"] = "vendor/openpi"
    reference = {"schema_version": 1, "checkpoint_sha256": checkpoint_hash,
                 "source_checkpoint_sha256": model_manifest["source_checkpoint_sha256"], "actions": actions,
                 "sample_observation_sha256": inventory["sample_observation.npz"]["sha256"],
                 "source_bundle_validation_sha256": inventory["bundle-validation.json"]["sha256"],
                 "robot_executed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for source, destination, digest in copies:
            _copy(source, stage / destination, digest)
        # The copied launcher defaults to this stable in-package profile path;
        # a custom source profile is normalized to it for Orin use.
        _write_json(stage / PROFILE_PATH, client_profile)
        _write_json(stage / "reference_actions.json", reference)
        payload = sorted(path for path in stage.rglob("*") if path.is_file())
        checksum_text = "".join(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in payload)
        _write_text(stage / "SHA256SUMS", checksum_text)
        files = [{"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(stage.rglob("*")) if path.is_file()]
        manifest = {
            "schema_version": 1, "status": "complete", "bundle_type": "airbot_robot_client", "client_only": True,
            "created_at": datetime.now(timezone.utc).isoformat(), "profile_id": profile["profile_id"],
            "policy_checkpoint_sha256": checkpoint_hash, "source_profile_sha256": profile_hash,
            "source_model": {"manifest_sha256": model_manifest_hash,
                             "validation_sha256": inventory["bundle-validation.json"]["sha256"],
                             "source_checkpoint_sha256": model_manifest["source_checkpoint_sha256"],
                             "scope": model_manifest["scope"]},
            "hardware_source_files": hardware,
            "copied_source_files": [{"source_path": str(source), "bundle_path": destination, "sha256": digest}
                                    for source, destination, digest in copies],
            "exporter_sha256": sha256(__file__), "files": files,
            "file_inventory_excludes": ["manifest.json"], "sha256sums_excludes": ["SHA256SUMS", "manifest.json"],
            "environment_bundled": False, "model_weights_bundled": False, "robot_executed": False,
        }
        _write_json(stage / "manifest.json", manifest)
        verified_inventory(stage, manifest)
        actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
        if actual != {item["path"] for item in files} | {"manifest.json"}:
            raise ValueError("Staged export contains unexpected files")
        for source, _, digest in copies:
            if sha256(source) != digest:
                raise ValueError(f"Source changed during export: {source}")
        if sha256(profile_source) != profile_hash or sha256(model_manifest_path) != model_manifest_hash:
            raise ValueError("Deployment profile or model manifest changed during export")
        if sha256(validation_path) != inventory["bundle-validation.json"]["sha256"]:
            raise ValueError("Model validation record changed during export")
        publish_new(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New robot-client directory; existing paths are refused")
    parser.add_argument("--openpi-root", help="Hardware sources; defaults to the profile's workstation OpenPI root")
    parser.add_argument("--profile", default=PROFILE_PATH,
                        help="Source deployment profile; custom can profiles are normalized in-package")
    parser.add_argument("--readme", help="Task-specific robot-client README")
    args = parser.parse_args(argv)
    manifest = export_robot(args.output, args.openpi_root, profile_path=args.profile, readme_path=args.readme)
    print(json.dumps({"status": "complete", "output": str(Path(args.output).resolve()),
                      "files": len(manifest["files"]), "policy_checkpoint_sha256": manifest["policy_checkpoint_sha256"],
                      "client_only": True, "robot_executed": False}))


if __name__ == "__main__":
    main()
