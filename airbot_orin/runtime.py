"""Validate a native runtime against source fixtures before serving its actions.

The original policy/transform and checkpoint are unchanged. Training provenance
stays in the original manifest; this adapter records target runtime provenance
separately and only bridges RGB to infer_depth after an explicit comparison.
No function in this module imports a robot SDK or sends a hardware command.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from airbot_depth.depth import DepthAnythingTransform
from airbot_depth.policy import AirbotDepthPolicy
from airbot_depth.serve import AirbotWebsocketAdapter, AirbotWebsocketServer


TOLERANCES = {
    "processor_atol": 1e-6,
    "mask_exact": True,
    "depth_max_abs": 0.005,
    "depth_mean_abs": 0.0005,
    "actions_refdepth_joints_atol": 0.0001,
    "actions_refdepth_gripper_atol": 0.00001,
    "actions_rgb_joints_atol": 0.002,
    "actions_rgb_gripper_atol": 0.0002,
    "rtol": 0,
}
CAMERA_KEYS = ["observation.images.head", "observation.images.wrist"]
LOGGER = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative package path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or escaping package file: {relative}")
    return path


def verify_package(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or not manifest.get("files"):
        raise ValueError("Orin package assembly is incomplete")
    seen = set()
    for item in manifest["files"]:
        if item["path"] in seen:
            raise ValueError("Duplicate package file identity")
        path = within(root, item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Package checksum mismatch: {item['path']}")
        seen.add(item["path"])
    required = {"model/policy.pt", "references/reference.json", "airbot_orin/runtime.py", "deploy.py"}
    if not required <= seen:
        raise ValueError("Orin package lacks its policy, fixtures or runtime")
    if (root / "airbot_orin/runtime.py").resolve() != Path(__file__).resolve():
        raise ValueError("The runtime must execute from the verified Orin package")
    for name in ("airbot_depth.depth", "airbot_depth.policy", "airbot_depth.model", "airbot_depth.serve",
                 "depth_policy.data", "depth_policy.model"):
        module_path = Path(sys.modules[name].__file__).resolve()
        expected = (root / "model" / (name.replace(".", "/") + ".py")).resolve()
        if module_path != expected:
            raise ValueError(f"Imported model source is outside the verified package: {name}")
    return manifest


def configure_runtime(device, threads=4):
    if threads < 1 or torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("Expected positive thread count and cpu/cuda device")
    if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("Expected CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Native CUDA Torch is unavailable; CPU fallback is not automatic")
        torch.ones(1, device=device).sum().item()


def environment(device):
    result = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_launcher": os.path.abspath(sys.executable),
        "python_prefix": sys.prefix,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_module_path": str(Path(torch.__file__).resolve()),
        "torch_module_sha256": sha256(torch.__file__),
        "torch_extension_sha256": sha256(torch._C.__file__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if str(device).startswith("cuda"):
        result.update(gpu=torch.cuda.get_device_name(device),
                      gpu_capability=list(torch.cuda.get_device_capability(device)))
    l4t = Path("/etc/nv_tegra_release")
    result["l4t"] = l4t.read_text().strip() if l4t.is_file() else None
    board = Path("/proc/device-tree/model")
    result["board_model"] = board.read_text().strip("\x00\n ") if board.is_file() else None
    return result


def load_reference(root):
    root = Path(root)
    reference = json.loads((root / "references/reference.json").read_text())
    if (reference.get("schema_version") != 1 or reference.get("status") != "complete"
            or reference.get("role") != "source_runtime_reference"
            or reference.get("camera_keys") != CAMERA_KEYS
            or reference.get("tolerances") != TOLERANCES):
        raise ValueError("Reference does not match the fixed source comparison protocol")
    samples = reference.get("samples", [])
    identities = [(item["episode_index"], item["frame_index"]) for item in samples]
    if len(samples) != 12 or len(set(identities)) != 12 or not any(item["split"] == "validation" for item in samples):
        raise ValueError("Reference must contain the twelve distinct audited frames, including held-out data")
    for item in samples:
        path = within(root / "references", item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Reference sample changed: {item['path']}")
    if reference["checkpoint_sha256"] != sha256(root / "model/policy.pt"):
        raise ValueError("Reference was generated for a different policy")
    if reference["bundle_manifest_sha256"] != sha256(root / "model/manifest.json"):
        raise ValueError("Reference belongs to a different immutable model bundle")
    model_manifest = json.loads((root / "model/manifest.json").read_text())
    expected_sources = {item["path"]: item["sha256"] for item in model_manifest["files"]
                        if item["path"].endswith(".py") and item["path"].split("/")[0] in {"airbot_depth", "depth_policy"}}
    if reference.get("source_files_sha256") != expected_sources:
        raise ValueError("Reference model-source fingerprints differ from the immutable bundle")
    for relative, digest in expected_sources.items():
        path = within(root / "model", relative)
        if sha256(path) != digest:
            raise ValueError(f"Reference model source changed: {relative}")
        name = relative[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-9]
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != path.resolve():
            raise ValueError(f"Imported model source differs from reference: {name}")
    return reference


def artifact_provenance_matches(training, actual):
    # Libraries are the only permitted provenance difference. Nothing is
    # overwritten to make the original infer_rgb strict comparison pass.
    return ({key: value for key, value in training.items() if key != "libraries"}
            == {key: value for key, value in actual.items() if key != "libraries"})


def load_components(root, reference, device):
    policy = AirbotDepthPolicy(Path(root) / "model/policy.pt", device=device, local_files_only=True)
    transform = DepthAnythingTransform(policy.depth_config, device=device, local_files_only=True)
    actual = transform.provenance()
    if (policy.camera_keys != CAMERA_KEYS or policy.prompt != reference["prompt"]
            or policy.depth_config != reference["depth_config"]
            or policy.depth_provenance != reference["training_depth_provenance"]
            or not artifact_provenance_matches(policy.depth_provenance, actual)):
        raise ValueError("Model files, transform source, processor or fixed depth configuration changed")
    return policy, transform, actual


def tensor_difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Compared tensors must have identical shapes and finite values")
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if not diff.size:
        raise ValueError("Compared tensors must be nonempty")
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def action_difference(actual, expected):
    if np.asarray(actual).shape != (8, 7) or np.asarray(expected).shape != (8, 7):
        raise ValueError("Expected the trained [8,7] action chunk")
    return {"joints_max_abs_rad": tensor_difference(actual[:, :6], expected[:, :6])["max_abs"],
            "gripper_max_abs_m": tensor_difference(actual[:, 6], expected[:, 6])["max_abs"]}


def comparison_passed(result):
    limits = [
        (value["max_abs"], TOLERANCES["processor_atol"]) for value in result["processor"]
    ] + [
        (result["depth_valid_pixels"]["max_abs"], TOLERANCES["depth_max_abs"]),
        (result["depth_valid_pixels"]["mean_abs"], TOLERANCES["depth_mean_abs"]),
        (result["actions_refdepth"]["joints_max_abs_rad"], TOLERANCES["actions_refdepth_joints_atol"]),
        (result["actions_refdepth"]["gripper_max_abs_m"], TOLERANCES["actions_refdepth_gripper_atol"]),
        (result["actions_rgb"]["joints_max_abs_rad"], TOLERANCES["actions_rgb_joints_atol"]),
        (result["actions_rgb"]["gripper_max_abs_m"], TOLERANCES["actions_rgb_gripper_atol"]),
    ]
    return (len(result["processor"]) == 2 and result["mask_exact"] is True
            and all(not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= limit
                    for value, limit in limits))


def compare_sample(policy, transform, fixture):
    observed_inputs = []
    def capture(module, arguments, kwargs):
        observed_inputs.append(kwargs["pixel_values"].detach().to("cpu", dtype=torch.float32).numpy().copy())
    hook = transform._model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        depth = transform([fixture[key] for key in CAMERA_KEYS])
    finally:
        hook.remove()
    if len(observed_inputs) != 2:
        raise ValueError("Depth transform did not process both original camera views")
    processors = [tensor_difference(value, fixture[key]) for value, key in
                  zip(observed_inputs, ["processor_head", "processor_wrist"], strict=True)]
    reference_depth = fixture["depth"]
    if depth.shape != (2, 2, 128, 128) or reference_depth.shape != depth.shape:
        raise ValueError("Depth fixture shape differs from the trained two-camera representation")
    mask_exact = bool(np.array_equal(depth[:, 1], reference_depth[:, 1]))
    valid = reference_depth[:, 1].astype(bool)
    depth_error = tensor_difference(depth[:, 0][valid], reference_depth[:, 0][valid])
    actions_depth = policy.infer_depth(fixture["state"], reference_depth, prompt=policy.prompt)
    actions_rgb = policy.infer_depth(fixture["state"], depth, prompt=policy.prompt)
    isolated = action_difference(actions_depth, fixture["actions_depth"])
    end_to_end = action_difference(actions_rgb, fixture["actions_rgb"])
    result = {"processor": processors, "mask_exact": mask_exact,
              "depth_valid_pixels": depth_error, "actions_refdepth": isolated, "actions_rgb": end_to_end}
    return {"passed": comparison_passed(result), **result}


def validate(root, device="cuda", threads=4):
    root = Path(root).resolve()
    verify_package(root)
    configure_runtime(device, threads)
    reference = load_reference(root)
    policy, transform, provenance = load_components(root, reference, device)
    comparisons = []
    for sample in reference["samples"]:
        with np.load(within(root / "references", sample["path"]), allow_pickle=False) as data:
            fixture = {key: data[key] for key in data.files}
        result = compare_sample(policy, transform, fixture)
        comparisons.append({"path": sample["path"], "sha256": sample["sha256"], **result})
        LOGGER.info("Reference %s: passed=%s, RGB action difference=%s", sample["path"], result["passed"], result["actions_rgb"])
    timings = []
    for _ in range(3):
        started = time.monotonic()
        depth = transform([fixture[key] for key in CAMERA_KEYS])
        policy.infer_depth(fixture["state"], depth, prompt=policy.prompt)
        timings.append((time.monotonic() - started) * 1000)
    runtime_environment = environment(device)
    return {
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in comparisons) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": policy.checkpoint_sha256,
        "package_manifest_sha256": sha256(root / "manifest.json"),
        "reference_sha256": sha256(root / "references/reference.json"),
        "runtime_source_sha256": sha256(__file__),
        "environment": runtime_environment,
        "training_depth_provenance": policy.depth_provenance,
        "runtime_depth_provenance": provenance,
        "tolerances": deepcopy(TOLERANCES),
        "comparisons": comparisons,
        "latency": {"samples_ms": timings, "mean_ms": float(np.mean(timings)),
                    "scope": "two native RGB arrays through depth and policy; excludes camera/robot I/O"},
        "orin_agx_runtime": (platform.machine() == "aarch64" and runtime_environment["l4t"] is not None
                             and "agx orin" in (runtime_environment["board_model"] or "").lower()),
        "robot_executed": False,
    }


def verify_attestation(root, path, device, threads):
    root, path = Path(root).resolve(), Path(path)
    verify_package(root)
    configure_runtime(device, threads)
    reference = load_reference(root)
    attestation = json.loads(path.read_text())
    expected = {"schema_version": 1, "status": "passed", "checkpoint_sha256": reference["checkpoint_sha256"],
                "package_manifest_sha256": sha256(root / "manifest.json"),
                "reference_sha256": sha256(root / "references/reference.json"),
                "runtime_source_sha256": sha256(__file__), "environment": environment(device),
                "tolerances": TOLERANCES, "robot_executed": False}
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise ValueError(f"Runtime validation is absent or stale: {key}; rerun validate in this environment")
    results = attestation.get("comparisons", [])
    if len(results) != len(reference["samples"]):
        raise ValueError("Attestation does not cover all reference samples")
    for actual, sample in zip(results, reference["samples"], strict=True):
        if (actual.get("path") != sample["path"] or actual.get("sha256") != sample["sha256"]
                or actual.get("passed") is not True or not comparison_passed(actual)):
            raise ValueError("Attestation sample identity or comparison failed")
    policy, transform, provenance = load_components(root, reference, device)
    if provenance != attestation.get("runtime_depth_provenance"):
        raise ValueError("Target runtime dependencies/model files changed since validation")
    if policy.depth_provenance != attestation.get("training_depth_provenance"):
        raise ValueError("Training provenance changed since validation")
    return policy, transform, attestation


class AttestedPolicy:
    """Expose the original policy contract with a separately validated RGB path."""

    def __init__(self, policy, transform):
        self.policy, self.transform = policy, transform
        for name in ("manifest", "camera_keys", "prompt", "fps", "model", "checkpoint_sha256"):
            setattr(self, name, getattr(policy, name))

    def infer_rgb(self, state, images, prompt=None):
        self.policy._validate_state_and_prompt(state, prompt)
        if not isinstance(images, dict) or set(images) != set(self.camera_keys):
            raise ValueError("Native RGB views must exactly match the trained cameras")
        for key, shape in zip(CAMERA_KEYS, [(1080, 1920, 3), (480, 848, 3)], strict=True):
            if not isinstance(images[key], np.ndarray) or images[key].shape != shape or images[key].dtype != np.uint8:
                raise ValueError(f"Expected original native RGB geometry for {key}: {shape}")
        depth = self.transform([images[key] for key in self.camera_keys])
        return self.policy.infer_depth(state, depth, prompt=prompt)


class AttestedAdapter(AirbotWebsocketAdapter):
    def __init__(self, policy, attestation_path, attestation):
        super().__init__(policy)
        self.identity = {"runtime_attestation_sha256": sha256(attestation_path),
                         "runtime_depth_provenance": deepcopy(attestation["runtime_depth_provenance"])}

    @property
    def metadata(self):
        return super().metadata | deepcopy(self.identity)


def serve(root, attestation_path, device="cuda", threads=4, port=8026):
    policy, transform, attestation = verify_attestation(root, attestation_path, device, threads)
    reference = load_reference(root)
    wrapped = AttestedPolicy(policy, transform)
    # Recheck one complete comparison and warm the GPU before accepting clients.
    with np.load(Path(root) / "references" / reference["samples"][0]["path"], allow_pickle=False) as sample:
        first = compare_sample(policy, transform, {key: sample[key] for key in sample.files})
        if not first["passed"]:
            raise ValueError("Startup reference comparison failed; repeat full native validation")
    adapter = AttestedAdapter(wrapped, attestation_path, attestation)
    asyncio.run(AirbotWebsocketServer(adapter, host="127.0.0.1", port=port).run())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "serve"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--port", type=int, default=8026)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "serve":
        if args.attestation is None:
            parser.error("serve requires --attestation")
        try:
            serve(args.root, args.attestation, args.device, args.threads, args.port)
        except KeyboardInterrupt:
            LOGGER.info("Local policy server stopped by operator")
            return 130
        return 0
    if args.output is None:
        parser.error("validate requires --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate(args.root, args.device, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "report": str(args.output.resolve()),
                      "orin_agx_runtime": result["orin_agx_runtime"], "robot_executed": False}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
