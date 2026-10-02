"""Assemble immutable models, robot client and references for local Orin deployment.

This is an artifact assembly step, not native installation or GPU validation.
Only bytes and manifests are inspected; no copied Python module is imported.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_airbot_robot import publish_new


MODEL_DIRECTORY = "deploy/airbot-paperbag200-da2-100k-20260924"
ROBOT_DIRECTORY = "deploy/airbot-paperbag200-da2-100k-robot-20260924"
OPENPI_CLIENT_SOURCE = Path("/home/user/wang-sm/openpi/packages/openpi-client/src/openpi_client")
POLICY_SHA256 = "8d09e474d3c9c0836a1f4896de05324b1fdaefc041a9d66a47caca116b17963d"
TEMPLATES = {
    "scripts/deploy_airbot_orin.py": "deploy.py",
    "scripts/prepare_airbot_orin_env.py": "prepare_env.py",
    "docs/airbot-paperbag-orin.md": "README.md",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or any(char in value for char in "\\\r\n\0"):
        raise ValueError("Inventory paths must be plain relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Inventory path escapes its source: {value!r}")
    return path.as_posix()


def _hash_value(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256 fingerprint")
    return value


def _source_file(root, relative):
    path = root / _relative(relative)
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlink is not an immutable source file: {path}")
        current = current.parent
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Source escapes its declared root: {path}")
    return path


def _all_files(root):
    result = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Source tree contains a symlink: {path}")
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Source tree contains a special file: {path}")
    return result


def _verify_sums(root, relative, expected):
    entries = {}
    for line in _source_file(root, relative).read_text().splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise ValueError(f"Invalid checksum listing: {root / relative}")
        digest, path = _hash_value(fields[0]), _relative(fields[1])
        if path in entries:
            raise ValueError(f"Duplicate checksum listing entry: {path}")
        entries[path] = digest
    if entries != expected:
        raise ValueError(f"Checksum listing differs from the complete inventory: {root / relative}")


def verify_bundle(directory):
    """Verify every recorded file and reject unrecorded additions before copying."""
    directory = Path(directory).resolve(strict=True)
    manifest_path = _source_file(directory, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Source bundle is not a complete schema-1 artifact: {directory}")
    inventory = {}
    for item in manifest.get("files", []):
        relative = _relative(item["path"])
        if relative == "manifest.json" or relative in inventory:
            raise ValueError(f"Duplicate or self-referential inventory entry: {relative}")
        path = _source_file(directory, relative)
        digest = _hash_value(item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError(f"Invalid file size: {relative}")
        if path.stat().st_size != item["bytes"] or sha256(path) != digest:
            raise ValueError(f"Source bundle checksum mismatch: {path}")
        inventory[relative] = digest
    if not inventory or _all_files(directory) != set(inventory) | {"manifest.json"}:
        raise ValueError(f"Source bundle has an empty or incomplete file inventory: {directory}")
    if "SHA256SUMS" in inventory:
        _verify_sums(directory, "SHA256SUMS", {key: value for key, value in inventory.items() if key != "SHA256SUMS"})
    return manifest, {**inventory, "manifest.json": sha256(manifest_path)}


def verify_references(directory, model_manifest, model_files, model_root, policy_sha256=None):
    directory = Path(directory).resolve(strict=True)
    policy_sha256 = POLICY_SHA256 if policy_sha256 is None else policy_sha256
    path = _source_file(directory, "reference.json")
    reference = json.loads(path.read_text())
    if (reference.get("schema_version") != 1 or reference.get("status") != "complete"
            or reference.get("role") != "source_runtime_reference"
            or reference.get("cross_platform_validated") is not False):
        raise ValueError("References must be complete source-runtime evidence, not native Orin validation")
    if (reference.get("checkpoint_sha256") != policy_sha256
            or reference.get("bundle_manifest_sha256") != model_files["manifest.json"]
            or reference.get("audit_sha256") != model_files["audit.json"]
            or model_manifest.get("audit_sha256") != model_files["audit.json"]):
        raise ValueError("Reference checkpoint, model bundle or audit identity mismatch")
    audit = json.loads(_source_file(model_root, "audit.json").read_text())
    if reference.get("dataset_manifest_sha256") != audit.get("manifest_sha256"):
        raise ValueError("Reference dataset identity differs from the training audit")
    samples = reference.get("samples", [])
    audited = audit.get("roundtrip", {}).get("frames", [])
    if len(samples) != 12 or len(audited) != 12:
        raise ValueError("The complete twelve-frame audited reference set is required")
    inventory = {"reference.json": sha256(path)}
    identities = set()
    for sample, frame in zip(samples, audited, strict=True):
        identity = sample.get("episode_index"), sample.get("frame_index")
        if identity in identities:
            raise ValueError("Reference sample identity is duplicated")
        identities.add(identity)
        for key in ("episode_index", "frame_index", "split", "timestamp", "camera_video_pts"):
            if sample.get(key) != frame.get(key):
                raise ValueError(f"Reference sample {key} differs from the audited frame")
        relative = _relative(sample["path"])
        if not relative.startswith("samples/") or not relative.endswith(".npz") or relative in inventory:
            raise ValueError("Reference samples require unique samples/*.npz paths")
        digest = _hash_value(sample["sha256"])
        if sha256(_source_file(directory, relative)) != digest:
            raise ValueError(f"Reference sample checksum mismatch: {relative}")
        inventory[relative] = digest
    _verify_sums(directory, "SHA256SUMS", inventory)
    inventory["SHA256SUMS"] = sha256(directory / "SHA256SUMS")
    if _all_files(directory) != set(inventory):
        raise ValueError("Reference directory contains files outside its recorded sample set")
    return reference, inventory


def _verify_client_init(path):
    """The two-file vendor package must have an inert version-only initializer."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for statement in tree.body:
        docstring = isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)
        version = (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                   and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "__version__"
                   and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str))
        if not (docstring or version):
            raise ValueError("Vendored openpi_client initializer must not import or execute extra runtime code")


def _copy(source, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256(destination) != digest:
        raise ValueError(f"Source changed during package assembly: {source}")


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def export_orin(output, references, *, root=ROOT, client_source=OPENPI_CLIENT_SOURCE, robot_bundle=None,
                model_directory=MODEL_DIRECTORY, robot_directory=ROBOT_DIRECTORY,
                policy_sha256=None, allow_nonformal=False, readme_path=None):
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to replace an existing Orin export: {output}")
    root = Path(root).resolve(strict=True)
    policy_sha256 = POLICY_SHA256 if policy_sha256 is None else policy_sha256
    if not isinstance(policy_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", policy_sha256) is None:
        raise ValueError("policy_sha256 must be a lowercase SHA256 fingerprint")
    model_root = (root / model_directory).resolve(strict=True)
    robot_root = (Path(robot_bundle).expanduser() if robot_bundle is not None else root / robot_directory).resolve(strict=True)
    reference_root = Path(references).expanduser().resolve(strict=True)
    client_source = Path(client_source).expanduser().resolve(strict=True)
    output = output.parent.resolve() / output.name
    for source in (model_root, robot_root, reference_root, client_source):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Orin output must be separate from every immutable source directory")
    model, model_files = verify_bundle(model_root)
    robot, robot_files = verify_bundle(robot_root)
    scope = model.get("scope", {})
    if (model.get("exported_checkpoint_sha256") != policy_sha256
            or model_files.get("policy.pt") != policy_sha256):
        raise ValueError("Model bundle checkpoint identity differs from the requested policy")
    if not allow_nonformal and (scope.get("formal_training_run") is not True
                                or scope.get("run_completed_steps") != 100000
                                or scope.get("checkpoint_training_step") != 76000):
        raise ValueError("Model is not the fixed best76k checkpoint from the formal 100k run")
    if allow_nonformal and (type(scope.get("run_completed_steps")) is not int
                            or scope.get("run_completed_steps") < 1
                            or type(scope.get("checkpoint_training_step")) is not int
                            or not 1 <= scope.get("checkpoint_training_step") <= scope.get("run_completed_steps")):
        raise ValueError("Non-formal model bundle has an invalid training scope")
    if not {"audit.json", "bundle-validation.json", "airbot_depth/policy.py", "airbot_depth/depth.py"} <= set(model_files):
        raise ValueError("Model bundle lacks its frozen runtime or audit artifacts")
    if (robot.get("bundle_type") != "airbot_robot_client" or robot.get("client_only") is not True
            or robot.get("policy_checkpoint_sha256") != policy_sha256
            or robot.get("source_model", {}).get("manifest_sha256") != model_files["manifest.json"]
            or robot["source_model"].get("validation_sha256") != model_files["bundle-validation.json"]):
        raise ValueError("Robot package does not belong to this exact model bundle")
    reference, reference_files = verify_references(reference_root, model, model_files, model_root,
                                                    policy_sha256=policy_sha256)
    sources = []
    for directory, prefix, inventory in ((model_root, "model", model_files), (robot_root, "robot", robot_files),
                                          (reference_root, "references", reference_files)):
        sources.extend((_source_file(directory, relative), f"{prefix}/{relative}", digest)
                       for relative, digest in sorted(inventory.items()))
    template_paths = dict(TEMPLATES)
    custom_readme = None
    if readme_path is not None:
        custom_readme = Path(readme_path).expanduser().resolve(strict=True)
        if not custom_readme.is_file():
            raise FileNotFoundError(custom_readme)
        template_paths.pop("docs/airbot-paperbag-orin.md", None)
    modules = sorted((root / "airbot_orin").glob("*.py"))
    if not modules or "__init__.py" not in {path.name for path in modules}:
        raise FileNotFoundError("The airbot_orin runtime package must exist before assembly")
    for path in modules:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", path.name) is None:
            raise ValueError("Orin runtime modules must have plain Python filenames")
        template_paths[f"airbot_orin/{path.name}"] = f"airbot_orin/{path.name}"
    for relative, destination in template_paths.items():
        source = _source_file(root, relative)
        sources.append((source, destination, sha256(source)))
    if custom_readme is not None:
        sources.append((custom_readme, "README.md", sha256(custom_readme)))
    initializer = _source_file(client_source, "__init__.py")
    _verify_client_init(initializer)
    for name in ("__init__.py", "msgpack_numpy.py"):
        source = _source_file(client_source, name)
        sources.append((source, f"vendor/openpi_client/{name}", sha256(source)))
    if len({destination for _, destination, _ in sources}) != len(sources):
        raise ValueError("Multiple sources map to the same package path")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for source, destination, digest in sources:
            _copy(source, stage / destination, digest)
        payload = {destination: digest for _, destination, digest in sources}
        _write_text(stage / "SHA256SUMS", "".join(f"{digest}  {path}\n" for path, digest in sorted(payload.items())))
        files = [{"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(stage.rglob("*")) if path.is_file()]
        manifest = {
            "schema_version": 1, "status": "complete", "bundle_type": "airbot_orin_local",
            "state": "awaiting_orin_installation", "cross_platform_validated": False,
            "inference_ready": False, "robot_executed": False, "environment_bundled": False,
            "created_at": datetime.now(timezone.utc).isoformat(), "policy_checkpoint_sha256": policy_sha256,
            "paths": {"model": "model", "robot": "robot", "references": "references", "runtime": "runtime", "vendor": "vendor"},
            "source_identity": {
                "model": {"path": str(model_root), "manifest_sha256": model_files["manifest.json"], "verified_files": len(model_files) - 1},
                "robot": {"path": str(robot_root), "manifest_sha256": robot_files["manifest.json"], "verified_files": len(robot_files) - 1},
                "references": {"path": str(reference_root), "reference_sha256": reference_files["reference.json"],
                               "sample_count": len(reference["samples"])},
                "copied_files": [{"source_path": str(source), "bundle_path": destination, "sha256": digest}
                                 for source, destination, digest in sources],
                "exporter_sha256": sha256(__file__),
            },
            "files": files, "file_inventory_excludes": ["manifest.json", "runtime/**"],
            "sha256sums_excludes": ["SHA256SUMS", "manifest.json", "runtime/**"],
            "assembly_scope": "Immutable artifact assembled; installation and native GPU validation are pending",
        }
        _write_text(stage / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        verify_bundle(stage)
        for source, _, digest in sources:
            if sha256(source) != digest:
                raise ValueError(f"Source changed during package assembly: {source}")
        publish_new(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New Orin package directory; existing paths are refused")
    parser.add_argument("--references", required=True, help="Completed source-runtime reference directory with twelve samples")
    parser.add_argument("--robot-bundle", help="Verified robot-client export for this same model; defaults to the original snapshot")
    parser.add_argument("--model-directory", default=MODEL_DIRECTORY,
                        help="Model bundle path relative to the repository root")
    parser.add_argument("--robot-directory", default=ROBOT_DIRECTORY,
                        help="Default robot-client bundle path relative to the repository root")
    parser.add_argument("--policy-sha256", help="Expected policy.pt SHA256; defaults to the historical paper-bag identity")
    parser.add_argument("--allow-nonformal", action="store_true",
                        help="Allow a completed subset such as the 100-episode can experiment")
    parser.add_argument("--readme", help="Optional task-specific README to place at the package root")
    args = parser.parse_args(argv)
    manifest = export_orin(args.output, args.references, robot_bundle=args.robot_bundle,
                           model_directory=args.model_directory, robot_directory=args.robot_directory,
                           policy_sha256=args.policy_sha256, allow_nonformal=args.allow_nonformal,
                           readme_path=args.readme)
    print(json.dumps({"status": manifest["status"], "state": manifest["state"],
                      "output": str(Path(args.output).resolve()), "files": len(manifest["files"]),
                      "policy_checkpoint_sha256": manifest["policy_checkpoint_sha256"], "inference_ready": False,
                      "cross_platform_validated": False, "robot_executed": False}))


if __name__ == "__main__":
    main()
