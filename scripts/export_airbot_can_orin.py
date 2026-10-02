#!/usr/bin/env python3
"""Build and check an AIRBOT can policy for an Orin deployment.

The historical paper-bag exporters intentionally pin one checkpoint and one
set of twelve reference fixtures.  This entry point keeps those artifacts
unchanged while accepting a new ``airbot_depth`` checkpoint.  It can:

* export a model-only bundle with the existing audited exporter;
* derive a deployment profile from the checkpoint/manifest rather than a
  copied paper-bag prompt or SHA256; and
* run an offline native-RGB -> Depth Anything -> action check without loading
  robot code or enabling hardware.

The optional ``reference`` and ``orin`` commands add the same layered target
runtime package used by the existing Orin deployment.  They require four
episodes selected by the source audit (three frames per episode, twelve NPZ
fixtures total).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TEMPLATE = ROOT / "configs/airbot_paperbag_deploy.json"
CAMERA_KEYS = ["observation.images.head", "observation.images.wrist"]
CAMERA_MAPPING = {
    "observation/base_0_rgb": CAMERA_KEYS[0],
    "observation/left_wrist_0_rgb": CAMERA_KEYS[1],
}
STATE_NAMES = [f"joint{index}.pos" for index in range(1, 7)] + ["eef.pos"]
UNITS = ["rad"] * 6 + ["m"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: str | Path, value: Any) -> None:
    path = Path(path).expanduser().absolute()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_checkpoint(value: str | Path, manifest: str | Path | None = None):
    """Return ``(policy.pt, bundle-directory, model-manifest)`` for a path."""
    path = Path(value).expanduser().resolve(strict=True)
    if path.is_dir():
        bundle = path
        checkpoint = bundle / "policy.pt"
    else:
        checkpoint = path
        bundle = path.parent if path.name == "policy.pt" else None
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {checkpoint}")
    manifest_path = Path(manifest).expanduser().resolve(strict=True) if manifest else None
    if manifest_path is None and bundle is not None and (bundle / "manifest.json").is_file():
        manifest_path = bundle / "manifest.json"
    model_manifest = _read_json(manifest_path) if manifest_path is not None else None
    if model_manifest is not None and model_manifest.get("status") == "complete":
        exported = model_manifest.get("exported_checkpoint_sha256")
        if exported and exported != sha256_file(checkpoint):
            raise ValueError("Model manifest and checkpoint SHA256 differ")
    return checkpoint, bundle, model_manifest, manifest_path


def checkpoint_identity(checkpoint: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the checkpoint contract without importing the model runtime."""
    import torch

    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_manifest = payload.get("manifest")
    if not isinstance(saved_manifest, dict) or saved_manifest.get("schema_version") != 1:
        raise ValueError("Checkpoint has no schema-1 dataset manifest")
    if manifest is not None and manifest.get("status") == "complete":
        expected = manifest.get("exported_checkpoint_sha256")
        if expected and expected != sha256_file(checkpoint):
            raise ValueError("Supplied model manifest does not identify this checkpoint")
    model_config = payload.get("model_config", {})
    training_config = payload.get("config", {})
    camera_keys = list(saved_manifest.get("camera_keys", []))
    if camera_keys != CAMERA_KEYS:
        raise ValueError(f"The Orin adapter supports exactly {CAMERA_KEYS}; checkpoint has {camera_keys}")
    state_dim, action_dim = saved_manifest.get("state_dim"), saved_manifest.get("action_dim")
    if state_dim != 7 or action_dim != 7:
        raise ValueError("Orin AIRBOT deployment requires seven state and action values")
    if saved_manifest.get("state_names") != STATE_NAMES or saved_manifest.get("action_names") != STATE_NAMES:
        raise ValueError("Checkpoint state/action ordering differs from the native AIRBOT contract")
    if saved_manifest.get("state_units") != UNITS or saved_manifest.get("action_units") != UNITS:
        raise ValueError("Checkpoint state/action units differ from the native AIRBOT contract")
    if saved_manifest.get("action_semantics") != "absolute_joint_position":
        raise ValueError("Orin deployment requires absolute native joint-position actions")
    prompt = saved_manifest.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Checkpoint manifest needs a nonempty task prompt")
    fps = saved_manifest.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Checkpoint manifest has an invalid camera/control FPS")
    horizon = training_config.get("horizon", model_config.get("horizon", 8))
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Checkpoint has an invalid action horizon")
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "prompt": prompt,
        "fps": float(fps),
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_names": list(STATE_NAMES),
        "action_names": list(STATE_NAMES),
        "state_units": list(UNITS),
        "action_units": list(UNITS),
        "action_semantics": "absolute_joint_position",
        "input_color": "RGB",
        "camera_keys": list(CAMERA_KEYS),
        "camera_mapping": deepcopy(CAMERA_MAPPING),
        "manifest": deepcopy(saved_manifest),
        "step": payload.get("step"),
    }


def _config_overrides(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = _read_json(path)
    # Experiment configs also contain training/data fields.  Keep only fields
    # that can affect the deployment profile; the checkpoint remains the
    # source of truth for prompt, dimensions and preprocessing.
    allowed = {"profile_id", "experiment_id", "prompt", "bundle", "inference_python",
               "robot_python", "openpi_root", "native_library_dir"}
    return {key: value[key] for key in allowed if key in value}


def make_profile(identity: dict[str, Any], bundle: str | Path, output: str | Path,
                 *, template: str | Path | None = None, config: str | Path | None = None) -> dict[str, Any]:
    """Create a profile accepted by ``airbot_deploy.client.load_profile``."""
    template_path = Path(template).expanduser().resolve() if template else DEFAULT_TEMPLATE
    profile = _read_json(template_path)
    overrides = _config_overrides(config)
    profile_id = overrides.get("profile_id", overrides.get("experiment_id"))
    if not profile_id:
        profile_id = f"airbot-can-{identity['checkpoint_sha256'][:12]}"
    profile["schema_version"] = 1
    profile["profile_id"] = str(profile_id)
    profile["bundle"] = str(Path(overrides.get("bundle", bundle)).expanduser().resolve())
    policy = profile.setdefault("policy", {})
    policy.update({key: deepcopy(identity[key]) for key in (
        "checkpoint_sha256", "prompt", "fps", "horizon", "state_dim", "action_dim",
        "state_names", "action_names", "state_units", "action_units", "action_semantics",
        "input_color", "camera_keys", "camera_mapping")})
    if "prompt" in overrides and overrides["prompt"] != identity["prompt"]:
        raise ValueError("Deployment config prompt differs from the trained checkpoint prompt")
    for environment in profile.setdefault("environments", {}).values():
        for field in ("inference_python", "robot_python", "openpi_root", "native_library_dir"):
            if field in overrides:
                environment[field] = overrides[field]
    profile["can_deployment"] = {
        "schema_version": 1,
        "source_checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_training_step": identity.get("step"),
        "model_manifest_sha256": sha256_file(Path(bundle) / "manifest.json")
        if (Path(bundle) / "manifest.json").is_file() else None,
        "robot_executed": False,
    }
    _write_new_json(output, profile)
    return profile


def verify_model_bundle(bundle: str | Path) -> dict[str, Any]:
    bundle = Path(bundle).expanduser().resolve(strict=True)
    manifest = _read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Model bundle is not a complete schema-1 artifact")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Model bundle inventory is empty")
    seen = set()
    for item in entries:
        relative = Path(item.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError(f"Invalid model bundle inventory path: {relative}")
        path = bundle / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing model bundle file: {relative}")
        if path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"Model bundle checksum mismatch: {relative}")
        seen.add(relative.as_posix())
    if not {"policy.pt", "manifest.json"} <= (seen | {"manifest.json"}):
        raise ValueError("Model bundle lacks policy.pt")
    if manifest.get("exported_checkpoint_sha256") != sha256_file(bundle / "policy.pt"):
        raise ValueError("Model bundle exported checkpoint identity is stale")
    return manifest


def command_bundle(args) -> dict[str, Any]:
    from scripts.export_airbot_depth import export_bundle

    checkpoint, _bundle, supplied_manifest, _manifest_path = resolve_checkpoint(args.checkpoint, args.manifest)
    result = export_bundle(checkpoint, args.audit, args.output, device=args.device, threads=args.threads)
    verify_model_bundle(args.output)
    checkpoint_path, bundle, model_manifest, _ = resolve_checkpoint(args.output)
    identity = checkpoint_identity(checkpoint_path, model_manifest)
    profile_path = Path(args.profile_output) if args.profile_output else Path(args.output).with_suffix(".profile.json")
    profile = make_profile(identity, bundle, profile_path, template=args.profile_template, config=args.config)
    result["generated_profile"] = str(profile_path.resolve())
    result["profile_id"] = profile["profile_id"]
    return result


def command_profile(args) -> dict[str, Any]:
    checkpoint, bundle, supplied_manifest, manifest_path = resolve_checkpoint(args.checkpoint, args.manifest)
    if bundle is None:
        raise ValueError("--checkpoint must point to a model bundle directory or policy.pt inside one")
    identity = checkpoint_identity(checkpoint, supplied_manifest)
    profile = make_profile(identity, bundle, args.output, template=args.profile_template, config=args.config)
    return {"profile": str(Path(args.output).resolve()), "profile_id": profile["profile_id"],
            "checkpoint_sha256": identity["checkpoint_sha256"], "manifest": str(manifest_path) if manifest_path else None}


def command_reference(args) -> dict[str, Any]:
    from scripts.export_airbot_orin_reference import export_reference

    indices = tuple(args.episode_indices)
    reference = export_reference(args.bundle, args.audit, args.output, args.device, args.threads,
                                episode_indices=indices, allow_nonformal=True)
    return {"reference": str(Path(args.output).resolve()), "samples": len(reference["samples"]),
            "checkpoint_sha256": reference["checkpoint_sha256"], "episode_indices": list(indices),
            "cross_platform_validated": False, "robot_executed": False}


def command_orin(args) -> dict[str, Any]:
    from scripts.export_airbot_orin import export_orin

    checkpoint, model_bundle, model_manifest, _ = resolve_checkpoint(args.model_bundle, args.manifest)
    if model_bundle is None:
        raise ValueError("--model-bundle must be a model bundle directory or policy.pt")
    policy_sha = sha256_file(checkpoint)
    # ``export_orin`` takes paths relative to root or absolute paths.  Passing
    # absolute paths lets this command assemble a package outside the source
    # repository without rewriting the historical constants.
    manifest = export_orin(args.output, args.references, root=ROOT,
                           client_source=args.client_source, robot_bundle=args.robot_bundle,
                           model_directory=str(model_bundle), robot_directory=str(args.robot_bundle),
                           policy_sha256=policy_sha, allow_nonformal=True, readme_path=args.readme)
    return {"output": str(Path(args.output).resolve()), "status": manifest["status"],
            "policy_checkpoint_sha256": policy_sha, "cross_platform_validated": False,
            "robot_executed": False}


def command_validate(args) -> dict[str, Any]:
    checkpoint, bundle, model_manifest, _ = resolve_checkpoint(args.checkpoint, args.manifest)
    if bundle is not None and (bundle / "hf_hub").is_dir():
        os.environ["HF_HUB_CACHE"] = str(bundle / "hf_hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from airbot_depth.policy import AirbotDepthPolicy

    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    policy = AirbotDepthPolicy(checkpoint, device=args.device, local_files_only=True)
    with np.load(args.observation, allow_pickle=False) as sample:
        fields = set(sample.files)
        expected = {"state", *policy.camera_keys}
        if fields != expected:
            raise ValueError(f"Observation must contain exactly {sorted(expected)}; got {sorted(fields)}")
        state = np.asarray(sample["state"])
        images = {key: np.asarray(sample[key]) for key in policy.camera_keys}
        for key, image in images.items():
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be a uint8 HWC RGB image")
        actions = policy.infer_rgb(state, images, prompt=policy.prompt)
    transform = policy._depth_transform
    if transform is None or transform.config != policy.depth_config:
        raise RuntimeError("RGB validation did not initialize the recorded Depth Anything transform")
    provenance = transform.provenance()
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_sha256": policy.checkpoint_sha256,
        "checkpoint_manifest_sha256": sha256_file(Path(bundle) / "manifest.json") if bundle else None,
        "prompt": policy.prompt,
        "camera_keys": list(policy.camera_keys),
        "rgb_shapes": {key: list(value.shape) for key, value in images.items()},
        "depth_shape": [len(policy.camera_keys), 2, policy.depth_config["image_size"],
                        policy.depth_config["image_size"]],
        "depth_config": policy.depth_config,
        "depth_provenance": provenance,
        "actions": np.asarray(actions, dtype=np.float32).tolist(),
        "action_shape": list(np.asarray(actions).shape),
        "device": str(policy.device),
        "robot_executed": False,
        "cross_platform_validated": False,
    }
    _write_new_json(args.output, report)
    return {"output": str(Path(args.output).resolve()), "status": report["status"],
            "checkpoint_sha256": report["checkpoint_sha256"], "action_shape": report["action_shape"],
            "robot_executed": False}


def command_inspect(args) -> dict[str, Any]:
    checkpoint, bundle, manifest, manifest_path = resolve_checkpoint(args.checkpoint, args.manifest)
    if bundle is None:
        raise ValueError("inspect expects a model bundle directory or policy.pt inside one")
    verified = verify_model_bundle(bundle)
    identity = checkpoint_identity(checkpoint, manifest)
    return {"bundle": str(bundle), "manifest": str(manifest_path), "status": verified["status"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_training_step": identity["step"], "robot_executed": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    bundle = commands.add_parser("bundle", help="Export a model-only bundle and dynamic can profile")
    bundle.add_argument("--checkpoint", required=True, help="Full audited training checkpoint")
    bundle.add_argument("--audit", required=True, help="Passed airbot_depth audit report")
    bundle.add_argument("--output", required=True, help="New model bundle directory")
    bundle.add_argument("--manifest", help="Optional manifest used to identify an external checkpoint")
    bundle.add_argument("--config", help="Optional can deployment JSON (profile_id/environment overrides)")
    bundle.add_argument("--profile-template", help="Profile template; defaults to the reviewed AIRBOT profile")
    bundle.add_argument("--profile-output", help="New generated profile path")
    bundle.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    bundle.add_argument("--threads", type=int, default=4)

    profile = commands.add_parser("profile", help="Derive a profile from an existing model bundle")
    profile.add_argument("--checkpoint", required=True, help="Model bundle directory or policy.pt")
    profile.add_argument("--manifest", help="Model bundle manifest.json")
    profile.add_argument("--output", required=True, help="New profile path")
    profile.add_argument("--config", help="Optional can deployment JSON")
    profile.add_argument("--profile-template", help="Profile template")

    reference = commands.add_parser("reference", help="Export twelve source-runtime fixtures for this model")
    reference.add_argument("--bundle", required=True, help="Completed model-only bundle")
    reference.add_argument("--audit", required=True, help="Exact audit bound to the model bundle")
    reference.add_argument("--output", required=True, help="New reference directory")
    reference.add_argument("--episode-indices", required=True, type=int, nargs=4,
                           help="Four distinct audited episodes (three frames each)")
    reference.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    reference.add_argument("--threads", type=int, default=4)

    orin = commands.add_parser("orin", help="Assemble the validated model+robot Orin package")
    orin.add_argument("--model-bundle", required=True, help="Model-only bundle directory")
    orin.add_argument("--references", required=True, help="Twelve-fixture source-runtime directory")
    orin.add_argument("--robot-bundle", required=True, help="Robot-client bundle for this checkpoint")
    orin.add_argument("--output", required=True, help="New Orin package directory")
    orin.add_argument("--manifest", help="Model manifest.json")
    orin.add_argument("--client-source", default="/home/user/wang-sm/openpi/packages/openpi-client/src/openpi_client",
                      help="openpi_client source directory to vendor")
    orin.add_argument("--readme", default=str(ROOT / "docs/airbot-can-orin.md"),
                      help="Task-specific README to place at the package root")

    validate = commands.add_parser("validate", help="Offline RGB -> Depth Anything -> action validation")
    validate.add_argument("--checkpoint", required=True, help="Model bundle directory or policy.pt")
    validate.add_argument("--manifest", help="Model bundle manifest.json")
    validate.add_argument("--observation", required=True, help="NPZ containing state and native RGB camera keys")
    validate.add_argument("--output", required=True, help="New JSON validation report")
    validate.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    validate.add_argument("--threads", type=int, default=4)

    inspect = commands.add_parser("inspect", help="Verify a model bundle inventory without loading weights")
    inspect.add_argument("--checkpoint", required=True, help="Model bundle directory or policy.pt")
    inspect.add_argument("--manifest", help="Model bundle manifest.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = {
        "bundle": command_bundle,
        "profile": command_profile,
        "reference": command_reference,
        "orin": command_orin,
        "validate": command_validate,
        "inspect": command_inspect,
    }[args.command](args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
