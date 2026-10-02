"""Export fixed source-runtime fixtures for an independent Orin compatibility check.

This runs the original, strictly checked RGB policy in its original environment.
It neither changes that policy's provenance rules nor certifies a target runtime.
The default keeps the historical formal paper-bag selection; ``--allow-nonformal``
and four explicit episode indices support a completed subset such as the can run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_airbot_robot import inside, publish_new, sha256, verified_inventory


CAMERA_KEYS = ("observation.images.head", "observation.images.wrist")
EPISODE_INDICES = (0, 100, 199, 5)
NPZ_KEYS = {"state", *CAMERA_KEYS, "depth", "actions_rgb", "actions_depth",
            "processor_head", "processor_wrist"}
SOURCE_ATOL = 1e-6
# Fixed before observing any target-platform errors. These are computation
# compatibility gates on fixtures, not robot motion limits or success claims.
TOLERANCES = {
    "rtol": 0.0,
    "processor_atol": 1e-6,
    "mask_exact": True,
    "depth_max_abs": 0.005,
    "depth_mean_abs": 0.0005,
    "actions_refdepth_joints_atol": 0.0001,
    "actions_refdepth_gripper_atol": 0.00001,
    "actions_rgb_joints_atol": 0.002,
    "actions_rgb_gripper_atol": 0.0002,
}


def array_record(key, value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("Reference arrays must never contain Python objects")
    return {"key": key, "shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def selected_frames(manifest, audit, episode_indices=None):
    """Require an ordered start/middle/end audit selection, with no resampling.

    The historical paper-bag export uses ``EPISODE_INDICES``.  New tasks may
    have fewer episodes, so callers can provide four episode indices explicitly
    while retaining the same twelve-frame (three frames per episode) contract.
    """
    episode_indices = tuple(EPISODE_INDICES if episode_indices is None else episode_indices)
    if len(episode_indices) != 4 or len(set(episode_indices)) != 4:
        raise ValueError("Reference export requires four distinct episode indices")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in episode_indices):
        raise ValueError("Episode indices must be integers")
    roundtrip = audit.get("roundtrip", {})
    if (roundtrip.get("passed") is not True
            or roundtrip.get("runtime_depth_provenance_exact_match") is not True
            or roundtrip.get("camera_field_order_verified") is not True
            or roundtrip.get("episode_indices") != list(episode_indices)
            or roundtrip.get("camera_keys") != list(CAMERA_KEYS)
            or roundtrip.get("absolute_tolerance") != SOURCE_ATOL):
        raise ValueError("Reference export requires the original passed 12-frame roundtrip audit")
    entries = {item["episode_index"]: item for item in manifest["episodes"]}
    expected = []
    for index in episode_indices:
        entry = entries[index]
        if entry["split"] not in {"train", "validation"}:
            raise ValueError("Reference split differs from the frozen audit selection")
        if type(entry["frames"]) is not int or entry["frames"] < 3:
            raise ValueError("Reference episodes must contain distinct start/middle/end frames")
        expected.extend((index, frame, entry["split"])
                        for frame in (0, entry["frames"] // 2, entry["frames"] - 1))
    if not any(split == "validation" for _, _, split in expected):
        raise ValueError("Reference selection must include at least one validation episode")
    frames = roundtrip.get("frames", [])
    if [(item.get("episode_index"), item.get("frame_index"), item.get("split"))
            for item in frames] != expected:
        raise ValueError("Reference frames must exactly match the 12 ordered audit frames")
    for item in frames:
        if (item.get("mask_exact_match") is not True
                or item.get("same_rgb_inputs_reused_for_both_paths") is not True
                or set(item.get("camera_rgb_inputs", {})) != set(CAMERA_KEYS)
                or set(item.get("camera_video_pts", {})) != set(CAMERA_KEYS)):
            raise ValueError("Roundtrip frame lacks exact RGB/mask/camera evidence")
        for name in ("depth_max_abs_difference", "action_chunk_max_abs_difference"):
            value = item.get(name, float("inf"))
            if not np.isfinite(value) or not 0 <= value <= SOURCE_ATOL:
                raise ValueError("Roundtrip frame exceeds the original source tolerance")
    return deepcopy(frames)


def _load_runtime(bundle, device, threads):
    # Hugging Face captures these settings at import time. Reject an ambiguous
    # in-process invocation instead of silently loading another model cache.
    if any(name == "transformers" or name.startswith("transformers.")
           or name == "huggingface_hub" or name.startswith("huggingface_hub.")
           for name in sys.modules):
        raise RuntimeError("Run reference export in a fresh Python process before Hugging Face imports")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.update(HF_HUB_CACHE=str(bundle / "hf_hub"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("Reference inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    import torch
    from airbot_depth.policy import AirbotDepthPolicy
    from scripts.export_airbot_depth import runtime_versions

    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    policy = AirbotDepthPolicy(bundle / "policy.pt", device=device, local_files_only=True)
    versions = runtime_versions(device)
    versions["runtime_flags"] = {
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    return policy, versions


def _load_source(root):
    from airbot_depth.lerobot import LeRobotSource
    return LeRobotSource(root)


def verify_reference_environment(actual, saved):
    for key in ("python", "machine", "packages", "cuda_runtime", "cudnn", "gpu"):
        if key not in actual or actual[key] != saved.get(key):
            raise ValueError(f"Reference runtime differs from original validated runtime: {key}")
    if actual["validation_device"].split(":")[0] != saved["validation_device"].split(":")[0]:
        raise ValueError("Reference device type differs from the original validated runtime")


def verify_runtime_sources(root, inventory):
    sources = {name: item["sha256"] for name, item in inventory.items()
               if name.endswith(".py") and name.split("/")[0] in {"airbot_depth", "depth_policy"}}
    required = {"airbot_depth/depth.py", "airbot_depth/model.py", "airbot_depth/policy.py",
                "depth_policy/model.py"}
    if not required <= set(sources):
        raise ValueError("Model bundle lacks frozen runtime source files")
    for relative, expected in sources.items():
        if sha256(inside(root, relative)) != expected:
            raise ValueError(f"Reference runtime source differs from model bundle: {relative}")
    for name, module in list(sys.modules.items()):
        relative = name.replace(".", "/") + ".py"
        if relative not in sources:
            continue
        path = Path(module.__file__).resolve(strict=True)
        if sha256(path) != sources[relative]:
            raise ValueError(f"Imported runtime source differs from model bundle: {name}")
    return sources


def capture_depth(transform, images):
    """Capture the actual FP32 tensors passed to DA without changing frozen code."""
    captured = []

    def capture(_module, _args, kwargs):
        values = kwargs["pixel_values"].detach().cpu().numpy().copy()
        if (values.dtype != np.float32 or values.ndim != 4 or values.shape[:2] != (1, 3)
                or min(values.shape[-2:]) < 14 or any(size % 14 for size in values.shape[-2:])
                or not np.isfinite(values).all()):
            raise ValueError("Unexpected actual Depth Anything processor input")
        captured.append(values)

    hook = transform._model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        depth = transform(images)
    finally:
        hook.remove()
    if len(captured) != len(CAMERA_KEYS):
        raise ValueError("Depth Anything must process exactly two independent camera inputs")
    return depth, captured


def infer_reference(policy, state, rgb, saved_depth):
    before = {key: array_record(key, value) for key, value in rgb.items()}
    actions_rgb = policy.infer_rgb(state, rgb, prompt=policy.prompt)
    transform = policy._depth_transform
    if transform.config != policy.depth_config or transform.provenance() != policy.depth_provenance:
        raise ValueError("Reference depth provenance differs from the original training conversion")
    depth, processor = capture_depth(transform, [rgb[key] for key in CAMERA_KEYS])
    actions_depth = policy.infer_depth(state, depth, prompt=policy.prompt)
    actions_saved = policy.infer_depth(state, saved_depth, prompt=policy.prompt)
    arrays = {"state": state, **rgb, "depth": depth, "actions_rgb": actions_rgb,
              "actions_depth": actions_depth, "processor_head": processor[0], "processor_wrist": processor[1]}
    expected_shapes = {"state": (7,), "depth": (2, 2, policy.depth_config["image_size"],
                                               policy.depth_config["image_size"]),
                       "actions_rgb": (8, 7), "actions_depth": (8, 7)}
    for name, shape in expected_shapes.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid reference array: {name}")
    if (np.any(depth < 0) or np.any(depth > 1) or not np.isin(depth[:, 1], [0, 1]).all()
            or np.any(depth[:, 0][depth[:, 1] == 0] != 0)):
        raise ValueError("Reference depth does not preserve the depth/validity contract")
    if not np.array_equal(depth[:, 1], saved_depth[:, 1]):
        raise ValueError("Reference validity mask differs from original converted depth")
    errors = {"depth_vs_converted_max_abs": float(np.max(np.abs(depth - saved_depth))),
              "actions_rgb_vs_depth_max_abs": float(np.max(np.abs(actions_rgb - actions_depth))),
              "actions_rgb_vs_converted_max_abs": float(np.max(np.abs(actions_rgb - actions_saved)))}
    if any(not np.isfinite(value) or value > SOURCE_ATOL for value in errors.values()):
        raise ValueError(f"Reference reproduction exceeds original source tolerance: {errors}")
    if any(array_record(key, rgb[key]) != before[key] for key in CAMERA_KEYS):
        raise ValueError("Reference inference mutated the original RGB inputs")
    return arrays, errors


def source_samples(source, data_dir, manifest, frames, episode_indices=None):
    import h5py

    entries = {item["episode_index"]: item for item in manifest["episodes"]}
    episode_indices = tuple(EPISODE_INDICES if episode_indices is None else episode_indices)
    for index in episode_indices:
        entry = entries[index]
        path = inside(data_dir, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Converted episode changed: {index}")
        original = source.load_episode(index)
        selected = [item for item in frames if item["episode_index"] == index]
        indices = [item["frame_index"] for item in selected]
        with h5py.File(path, "r") as stream:
            stamps = stream["timestamp"][indices]
            if (not np.array_equal(stamps, original["timestamp"][indices])
                    or not np.array_equal(stream["state"][indices], original["state"][indices])):
                raise ValueError("Source state/timestamps differ from the converted dataset")
            views = {key: list(source.rgb_frames(index, key, stamps)) for key in CAMERA_KEYS}
            if any(len(values) != len(indices) for values in views.values()):
                raise ValueError("Source video decoder returned the wrong frame count")
            for offset, item in enumerate(selected):
                if float(stamps[offset]) != item["timestamp"]:
                    raise ValueError("Source timestamp differs from original roundtrip audit")
                rgb = {key: views[key][offset][0] for key in CAMERA_KEYS}
                for key, value in rgb.items():
                    identity = array_record(key, value)
                    identity.pop("key")
                    if (identity != item["camera_rgb_inputs"][key]
                            or float(views[key][offset][1]) != item["camera_video_pts"][key]):
                        raise ValueError(f"Source RGB/PTS differs from original audit: {index}/{key}")
                    if value.dtype != np.uint8 or list(value.shape) != source.info["features"][key]["shape"]:
                        raise ValueError("Source RGB geometry differs from native dataset metadata")
                yield item, stream["state"][item["frame_index"]], rgb, stream["depth"][item["frame_index"]]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Converted episode changed during reference inference: {index}")


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def export_reference(bundle, audit_path, output, device="cuda", threads=4, *, root=ROOT,
                     episode_indices=None, allow_nonformal=False):
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite reference export: {output}")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    bundle, root = Path(bundle).resolve(strict=True), Path(root).resolve(strict=True)
    audit_path = Path(audit_path).resolve(strict=True)
    output = output.parent.resolve() / output.name
    if output.is_relative_to(bundle):
        raise ValueError("Reference output must be separate from the immutable model bundle")
    bundle_manifest_path = inside(bundle, "manifest.json")
    bundle_manifest_sha = sha256(bundle_manifest_path)
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    inventory = verified_inventory(bundle, bundle_manifest)
    required = {"policy.pt", "audit.json", "runtime-versions.json", "bundle-validation.json"}
    if not required <= set(inventory):
        raise ValueError("Model bundle lacks its checkpoint, audit, validation or environment")
    scope = bundle_manifest.get("scope", {})
    if not allow_nonformal:
        if (scope.get("formal_training_run") is not True
                or scope.get("run_completed_steps") != 100000):
            raise ValueError("Reference export requires the formal 100000-step model bundle")
    elif (type(scope.get("run_completed_steps")) is not int
          or scope.get("run_completed_steps") < 1):
        raise ValueError("Non-formal reference export requires a completed training-step count")
    audit_sha = sha256(audit_path)
    if audit_sha != inventory["audit.json"]["sha256"] or audit_sha != bundle_manifest.get("audit_sha256"):
        raise ValueError("Supplied audit is not the exact audit bound to this model bundle")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema_version") != 1 or audit.get("status") != "passed"
            or audit.get("roundtrip", {}).get("checkpoint_sha256") != bundle_manifest["source_checkpoint_sha256"]):
        raise ValueError("Audit does not identify the bundle's roundtrip-tested checkpoint")
    if allow_nonformal:
        if audit.get("training_complete_steps") != scope.get("run_completed_steps"):
            raise ValueError("Audit training steps differ from the model bundle scope")
    elif (audit.get("formal_100000_step_training") is not True
          or audit.get("training_complete_steps") != 100000):
        raise ValueError("Audit does not identify the bundle's formal 100000-step run")
    checkpoint_sha = inventory["policy.pt"]["sha256"]
    validation = json.loads(inside(bundle, "bundle-validation.json").read_text())
    if (bundle_manifest.get("exported_checkpoint_sha256") != checkpoint_sha
            or validation.get("status") != "passed"
            or validation.get("exported_checkpoint_sha256") != checkpoint_sha
            or validation.get("source_checkpoint_sha256") != bundle_manifest["source_checkpoint_sha256"]):
        raise ValueError("Bundle checkpoint identity differs from its independent validation")
    data_dir = Path(audit["data"]).resolve(strict=True)
    data_manifest_path = inside(data_dir, "manifest.json")
    if sha256(data_manifest_path) != audit["manifest_sha256"]:
        raise ValueError("Current converted dataset manifest differs from the audited manifest")
    manifest = json.loads(data_manifest_path.read_text())
    episode_indices = tuple(EPISODE_INDICES if episode_indices is None else episode_indices)
    frames = selected_frames(manifest, audit, episode_indices)
    if output.is_relative_to(data_dir) or output.is_relative_to(Path(manifest["source"]["root"]).resolve()):
        raise ValueError("Reference output must be separate from source and converted datasets")
    sources = verify_runtime_sources(root, inventory)
    source = _load_source(manifest["source"]["root"])
    if (source.camera_keys != list(CAMERA_KEYS) or source.camera_keys != manifest["camera_keys"]
            or source.prompt != manifest["prompt"]):
        raise ValueError("Source and converted dataset camera/prompt contracts differ")
    source_fingerprint = source.fingerprint(list(episode_indices))
    original_files = {item["path"]: item for item in manifest["source"]["files"]}
    if any(item != original_files.get(item["path"]) for item in source_fingerprint["files"]):
        raise ValueError("Reference source files differ from the original conversion fingerprint")
    policy, versions = _load_runtime(bundle, device, threads)
    verify_reference_environment(versions, json.loads(inside(bundle, "runtime-versions.json").read_text()))
    if (policy.checkpoint_sha256 != checkpoint_sha or policy.manifest != manifest
            or policy.camera_keys != list(CAMERA_KEYS)):
        raise ValueError("Loaded reference policy differs from the frozen manifest/checkpoint")
    verify_runtime_sources(root, inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        (stage / "samples").mkdir()
        samples = []
        for item, state, rgb, saved_depth in source_samples(source, data_dir, manifest, frames, episode_indices):
            arrays, errors = infer_reference(policy, state, rgb, saved_depth)
            relative = f"samples/episode_{item['episode_index']:06d}_frame_{item['frame_index']:06d}.npz"
            with (stage / relative).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            with np.load(stage / relative, allow_pickle=False) as saved:
                if set(saved.files) != NPZ_KEYS or any(not np.array_equal(saved[key], value)
                                                     for key, value in arrays.items()):
                    raise ValueError("Saved reference NPZ differs from inference arrays")
            samples.append({"path": relative, "sha256": sha256(stage / relative),
                            "episode_index": item["episode_index"], "frame_index": item["frame_index"],
                            "split": item["split"], "timestamp": item["timestamp"],
                            "camera_video_pts": item["camera_video_pts"],
                            "arrays": [array_record(key, value) for key, value in arrays.items()],
                            "source_reproduction": errors})
            print(json.dumps({"stage": "reference_sample", "completed": len(samples),
                              "total": 12, "episode_index": item["episode_index"],
                              "frame_index": item["frame_index"], **errors}), flush=True)
        if len(samples) != 12:
            raise ValueError("Reference export did not produce all twelve audited frames")
        verified_inventory(bundle, bundle_manifest)
        if (sha256(bundle_manifest_path) != bundle_manifest_sha or sha256(audit_path) != audit_sha
                or sha256(data_manifest_path) != audit["manifest_sha256"]
                or verify_runtime_sources(root, inventory) != sources):
            raise ValueError("Frozen inputs changed while building reference fixtures")
        reference = {
            "schema_version": 1, "status": "complete", "role": "source_runtime_reference",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint_sha,
            "source_checkpoint_sha256": bundle_manifest["source_checkpoint_sha256"],
            "checkpoint_training_step": bundle_manifest["scope"]["checkpoint_training_step"],
            "bundle_manifest_sha256": bundle_manifest_sha, "audit_sha256": audit_sha,
            "dataset_manifest_sha256": audit["manifest_sha256"],
            "depth_config": policy.depth_config, "training_depth_provenance": policy.depth_provenance,
            "source_files_sha256": sources, "runtime_versions": versions,
            "exporter_sha256": sha256(__file__),
            "reader_source_sha256": sha256(inside(root, "airbot_depth/lerobot.py")),
            "source_fingerprint": source_fingerprint,
            "episode_indices": list(episode_indices),
            "camera_keys": list(CAMERA_KEYS),
            "camera_shapes": {key: source.info["features"][key]["shape"] for key in CAMERA_KEYS},
            "prompt": policy.prompt, "state_names": manifest["state_names"],
            "action_names": manifest["action_names"], "action_units": manifest["action_units"],
            "action_semantics": manifest["action_semantics"],
            "dataset_fps": manifest["fps"], "horizon": 8,
            "depth_array_stage": "float32_normalized_depth_and_mask_before_policy_float16_rounding",
            "processor_array_stage": "actual_float32_pixel_values_entering_depth_model_forward",
            "tolerances": deepcopy(TOLERANCES), "tolerances_fixed_before_target_measurement": True,
            "source_reproduction_atol": SOURCE_ATOL, "samples": samples,
            "cross_platform_validated": False, "robot_executed": False,
            "validation_scope": "Source-runtime fixture export only. Target hardware must independently "
                                "pass every layered comparison with these fixed tolerances before "
                                "issuing a runtime attestation. No robot success rate is measured.",
        }
        _write_json(stage / "reference.json", reference)
        with (stage / "SHA256SUMS").open("x", encoding="utf-8") as stream:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS":
                    stream.write(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_new(stage, output)
        return reference
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Immutable model bundle")
    parser.add_argument("--audit", required=True, help="Exact audit report bound to the model bundle")
    parser.add_argument("--output", required=True, help="New separate reference directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--episode-indices", type=int, nargs=4,
                        help="Four audited episode indices; defaults to the historical 0,100,199,5 set")
    parser.add_argument("--allow-nonformal", action="store_true",
                        help="Allow completed subsets such as the 100-episode can experiment")
    args = parser.parse_args(argv)
    reference = export_reference(args.bundle, args.audit, args.output, args.device, args.threads,
                                 episode_indices=args.episode_indices, allow_nonformal=args.allow_nonformal)
    print(json.dumps({"status": reference["status"], "output": str(Path(args.output).resolve()),
                      "samples": len(reference["samples"]), "cross_platform_validated": False}), flush=True)


if __name__ == "__main__":
    main()
