"""Export and independently validate a portable AIRBOT depth inference directory.

Only a checkpoint identified by a passed offline audit may be exported. A
compatible Python environment remains an external dependency; this script
copies neither a virtual environment nor robot-control software.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airbot_depth.common import atomic_json, sha256_file


RUNTIME_FILES = tuple(
    [f"airbot_depth/{name}.py" for name in ("__init__", "common", "depth", "model", "policy", "serve")]
    + [f"depth_policy/{name}.py" for name in ("__init__", "common", "data", "episodes", "model")]
)
CORE_FILES = ("airbot_depth/depth.py", "airbot_depth/model.py", "depth_policy/model.py")
INFERENCE_KEYS = (
    "schema_version", "format", "config", "image_input_precision", "model_config", "model",
    "statistics", "vocabulary", "depth_config", "depth_provenance", "action_semantics",
    "manifest_sha256", "manifest", "step",
)
HF_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
DEPENDENCIES = (
    "numpy", "torch", "torchvision", "transformers", "pillow", "huggingface-hub",
    "h5py", "safetensors", "tokenizers", "openpi-client", "msgpack", "websockets", "typing-extensions",
)


def continuation_checkpoint_origin(payload, audit, checkpoint_sha256):
    chain = audit.get("continuation", {})
    restoration, metrics = chain.get("restoration", {}), chain.get("metrics", {})
    final_optimizer = chain.get("final_optimizer", {})
    if (chain.get("schema_version") != 1 or chain.get("verified") is not True
            or chain.get("start_step") != 50000 or chain.get("end_step") != 100000
            or chain.get("parent_run_target_steps") != 50000
            or chain.get("initialization") != "resume_full_training_state"
            or chain.get("root_initialization") != "all_policy_parameters_random"
            or restoration.get("all_restored_exact") is not True
            or restoration.get("first_batch_recomputed") is not True
            or restoration.get("runtime_flags_restored") is not True
            or metrics.get("exact_parent_prefix") is not True
            or metrics.get("logged_step_coverage_verified") is not True
            or final_optimizer.get("min") != 100000 or final_optimizer.get("max") != 100000
            or not final_optimizer.get("count")):
        raise ValueError("100000-step export requires the verified full-state parent continuation chain")
    identities = [item for item in chain.get("checkpoint_origins", [])
                  if item.get("checkpoint_sha256") == checkpoint_sha256]
    if not identities or any(item.get("step") != payload["step"] for item in identities):
        raise ValueError("Checkpoint has no matching origin in the verified continuation chain")
    origins = {(item.get("phase"), item.get("origin_run_target_steps"), item.get("origin_run_sha256"))
               for item in identities}
    if len(origins) != 1:
        raise ValueError("Continuation audit assigns conflicting checkpoint origins")
    phase, target, origin_sha = origins.pop()
    if phase == "parent":
        inherited = {item["sha256"]: item["step"] for item in chain.get("inherited_checkpoints", [])}
        if (target != 50000 or payload["step"] > 50000 or origin_sha != chain.get("parent_run_sha256")
                or inherited.get(checkpoint_sha256) != payload["step"]):
            raise ValueError("Parent checkpoint is not bound to the verified original run and inherited bytes")
    elif phase == "continuation":
        if (target != 100000 or not 50000 < payload["step"] <= 100000 or origin_sha != audit.get("run_sha256")
                or payload.get("initialization") != chain["initialization"]
                or payload.get("root_initialization") != chain["root_initialization"]
                or payload.get("resume", {}).get("parent_run_sha256") != chain.get("parent_run_sha256")
                or payload.get("resume", {}).get("parent_checkpoint_sha256") != chain.get("parent_checkpoint_sha256")):
            raise ValueError("New checkpoint is not bound to the verified continuation run")
    else:
        raise ValueError("Unknown checkpoint continuation phase")
    if payload["config"].get("steps") != target:
        raise ValueError("Checkpoint origin target contradicts its original saved config.steps")
    return {"checkpoint_origin_phase": phase, "checkpoint_origin_run_target_steps": target,
            "checkpoint_origin_run_sha256": origin_sha}


def verify_audit_claims(payload, audit, checkpoint_sha256):
    """Tie the audit to this checkpoint, and distinguish run completion from step."""
    if audit.get("schema_version") != 1 or audit.get("status") != "passed":
        raise ValueError("Export requires a passed schema-1 Airbot audit report")
    if audit.get("manifest_sha256") != payload.get("manifest_sha256"):
        raise ValueError("Audit and checkpoint use different dataset manifests")
    candidates = [item for item in audit.get("validation", {}).get("checkpoints", [])
                  if item.get("checkpoint_sha256") == checkpoint_sha256]
    roundtrip = audit.get("roundtrip", {})
    roundtrip_matches = roundtrip.get("checkpoint_sha256") == checkpoint_sha256
    if not candidates and not roundtrip_matches:
        raise ValueError("Checkpoint SHA256 was neither evaluated nor roundtrip-tested in this audit")
    if any(item.get("step") != payload.get("step") for item in candidates):
        raise ValueError("Audit checkpoint step differs from the checkpoint payload")
    if roundtrip.get("passed") is not True or roundtrip.get("runtime_depth_provenance_exact_match") is not True:
        raise ValueError("Audit lacks a passed offline/online depth-provenance roundtrip")
    completed = audit.get("training_complete_steps")
    if (type(completed) is not int or completed < 1
            or type(payload.get("step")) is not int or not 1 <= payload["step"] <= completed):
        raise ValueError("Audit and checkpoint training-step records are inconsistent")
    if audit.get("formal_50000_step_training") is not (completed == 50000):
        raise ValueError("Audit's 50000-step label contradicts its completed-step count")
    if audit.get("formal_100000_step_training", False) is not (completed == 100000):
        raise ValueError("Audit's 100000-step label contradicts its completed-step count")
    if completed == 100000:
        origin = continuation_checkpoint_origin(payload, audit, checkpoint_sha256)
    else:
        if payload["config"].get("steps") != completed:
            raise ValueError("Audit and checkpoint training-step records are inconsistent")
        origin = {"checkpoint_origin_phase": "original",
                  "checkpoint_origin_run_target_steps": payload["config"]["steps"],
                  "checkpoint_origin_run_sha256": audit.get("run_sha256")}
    dataset = audit["dataset"]
    training, validation = dataset["train_episode_indices"], dataset["validation_episode_indices"]
    selected = payload["manifest"]["source"]["selected_episode_indices"]
    if (len(set(training + validation)) != len(training + validation)
            or sorted(training + validation) != sorted(selected)
            or dataset["train_episodes"] != len(training) or dataset["validation_episodes"] != len(validation)):
        raise ValueError("Audited episode split differs from the checkpoint dataset")
    full_dataset = (len(selected) == payload["manifest"]["source"]["total_episodes"] == 200
                    and len(training) == 180 and len(validation) == 20)
    if dataset.get("formal_200_episode_dataset") is True and not full_dataset:
        raise ValueError("Audit's formal dataset label contradicts its episode split")
    formal = (completed in (50000, 100000) and full_dataset
              and dataset.get("formal_200_episode_dataset") is True
              and dataset.get("formal_180_train_20_validation") is True)
    if formal and (audit["source_code"].get("formal_strict_source_equality") is not True
                   or audit["source_code"].get("all_saved_files_unchanged") is not True):
        raise ValueError("Formal export requires the strict unchanged-source audit")
    return {
        "role": f"formal_{completed}_step_training_run" if formal else "smoke_or_partial_training_run",
        "formal_training_run": formal,
        "checkpoint_training_step": payload["step"],
        "run_completed_steps": completed,
        "train_episodes": len(training), "validation_episodes": len(validation),
        "checkpoint_evaluated": bool(candidates), "checkpoint_roundtrip_tested": roundtrip_matches,
        **origin,
        "robot_executed": False, "real_robot_success_rate_measured": False,
    }


def verify_runtime_sources(payload, audit, root=ROOT):
    """Use audited runtime bytes; core depth/model bytes must also match training."""
    root = Path(root)
    records = audit.get("source_code", {}).get("files", [])
    if audit.get("source_code", {}).get("depth_and_policy_model_unchanged") is not True:
        raise ValueError("Audit did not establish unchanged depth and policy model implementations")
    result = []
    for relative in RUNTIME_FILES:
        current = sha256_file(root / relative)
        audited = [item for item in records if item.get("path") == relative]
        if any(item.get("current_sha256") != current for item in audited):
            raise ValueError(f"Runtime source changed after the audit: {relative}")
        if relative in CORE_FILES:
            if not audited or payload.get("source", {}).get(relative) != current:
                raise ValueError(f"Core inference source differs from the audited training source: {relative}")
            if any(item.get("saved_sha256") != current for item in audited):
                raise ValueError(f"Core inference source differs from the saved audit source: {relative}")
        if relative == "airbot_depth/policy.py" and not audited:
            raise ValueError("The inference policy must have an audited source hash")
        result.append({"path": relative, "sha256": current,
                       "present_in_training_audit": bool(audited)})
    if payload["depth_provenance"].get("implementation_sha256") != sha256_file(root / "airbot_depth/depth.py"):
        raise ValueError("Depth preprocessing provenance differs from the runtime source")
    return result


def inference_payload(payload, checkpoint_sha256, scope):
    missing = set(INFERENCE_KEYS) - set(payload)
    if missing:
        raise ValueError(f"Checkpoint lacks required inference fields: {sorted(missing)}")
    result = {name: payload[name] for name in INFERENCE_KEYS}
    result["inference_export"] = {"schema_version": 1,
                                  "source_checkpoint_sha256": checkpoint_sha256,
                                  "scope": deepcopy(scope),
                                  "training_state_removed": sorted(set(payload) - set(INFERENCE_KEYS))}
    return result


def resolve_hf_artifacts(payload):
    from airbot_depth.depth import canonical_depth_config
    from transformers.utils.hub import cached_file

    config = payload["depth_config"]
    if canonical_depth_config(config) != config:
        raise ValueError("Checkpoint depth configuration is not canonical")
    expected = payload["depth_provenance"]["files_sha256"]
    if set(expected) != set(HF_FILES):
        raise ValueError("Expected the three pinned Depth Anything HF artifacts")
    result = {}
    for name in HF_FILES:
        path = Path(cached_file(config["model_id"], name, revision=config["revision"], local_files_only=True))
        if sha256_file(path) != expected[name]:
            raise ValueError(f"Cached Depth Anything artifact differs from provenance: {name}")
        result[name] = path
    return result


def runtime_versions(device):
    return {
        "schema_version": 1, "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {name: version(name) for name in DEPENDENCIES},
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "validation_device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "environment_bundled": False, "cross_platform_validated": False,
        "external_dependencies": ["Compatible installed Python environment", "openpi_client package"],
    }


# Executed by a separate interpreter with cwd/cache set to the exported bundle.
# Deliberately imports no exporter or source-repository utility.
CHILD_VALIDATION = r'''
import json
from pathlib import Path
import sys
import numpy as np
import torch

bundle = Path.cwd().resolve()
original_root = Path(sys.argv[1]).resolve()
device, threads = sys.argv[2], int(sys.argv[3])
torch.set_num_threads(threads)
for entry in sys.path:
    location = Path(entry or '.').resolve()
    if location.is_relative_to(original_root) and not location.is_relative_to(bundle):
        raise RuntimeError(f'Original repository leaked into subprocess sys.path: {location}')
from airbot_depth.policy import AirbotDepthPolicy
policy = AirbotDepthPolicy(bundle / 'policy.pt', device=device, local_files_only=True)
with np.load(bundle / 'sample_observation.npz', allow_pickle=False) as sample:
    if set(sample.files) != {'state', *policy.camera_keys}:
        raise ValueError('Sample must contain native state and exactly the RGB camera keys')
    actions = policy.infer_rgb(sample['state'], {key: sample[key] for key in policy.camera_keys},
                               prompt=policy.prompt)
origins = {}
for name, module in list(sys.modules.items()):
    if name == 'airbot_depth' or name.startswith('airbot_depth.') or name == 'depth_policy' or name.startswith('depth_policy.'):
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(bundle):
            raise RuntimeError(f'Runtime import did not come from bundle: {name}: {location}')
        origins[name] = str(location.relative_to(bundle))
from transformers.utils.hub import cached_file
cached_artifacts = {}
for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
    location = Path(cached_file(policy.depth_config['model_id'], name,
                               revision=policy.depth_config['revision'], local_files_only=True)).resolve()
    if not location.is_relative_to(bundle / 'hf_hub'):
        raise RuntimeError(f'Depth model artifact did not come from bundle: {location}')
    cached_artifacts[name] = str(location.relative_to(bundle))
print(json.dumps({'actions': actions.tolist(), 'shape': list(actions.shape), 'dtype': str(actions.dtype),
                  'finite': bool(np.isfinite(actions).all()), 'module_origins': origins,
                  'cached_artifacts': cached_artifacts, 'checkpoint_sha256': policy.checkpoint_sha256,
                  'depth_provenance_matches': policy._depth_transform.provenance() == policy.depth_provenance,
                  'original_source_in_sys_path': False, 'robot_executed': False}, allow_nan=False))
'''


def validate_bundle(output, original_checkpoint, device, threads):
    from airbot_depth.policy import AirbotDepthPolicy

    policy = AirbotDepthPolicy(original_checkpoint, device=device, local_files_only=True)
    with np.load(output / "sample_observation.npz", allow_pickle=False) as sample:
        if set(sample.files) != {"state", *policy.camera_keys}:
            raise ValueError("Audited sample must contain native state and exactly the RGB camera keys")
        reference = policy.infer_rgb(sample["state"], {key: sample[key] for key in policy.camera_keys},
                                     prompt=policy.prompt)
    if reference.shape != (policy.model.horizon, 7) or not np.isfinite(reference).all():
        raise ValueError("Original checkpoint produced an invalid reference action")
    source_sha256 = policy.checkpoint_sha256
    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "TRANSFORMERS_CACHE"):
        environment.pop(name, None)
    environment.update(HF_HUB_CACHE=str(output / "hf_hub"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_VALIDATION, str(ROOT), device, str(threads)],
        cwd=output, env=environment, capture_output=True, text=True, timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(f"Isolated bundle inference failed:\n{completed.stderr[-12000:]}\n{completed.stdout[-2000:]}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    actions = np.asarray(result.pop("actions"), dtype=np.float32)
    if actions.shape != reference.shape or not np.isfinite(actions).all():
        raise ValueError("Isolated bundle produced an invalid action array")
    np.testing.assert_allclose(actions, reference, rtol=0, atol=1e-6)
    if result["checkpoint_sha256"] != sha256_file(output / "policy.pt") or not result["depth_provenance_matches"]:
        raise ValueError("Isolated runtime identity differs from the exported artifacts")
    return {
        "schema_version": 1, "status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": source_sha256,
        "exported_checkpoint_sha256": result.pop("checkpoint_sha256"),
        "action_max_absolute_difference": float(np.max(np.abs(actions - reference))),
        "action_all_values_exact": bool(np.array_equal(actions, reference)),
        "atol": 1e-6, "rtol": 0, "reference_actions": reference.tolist(), "bundle_actions": actions.tolist(),
        "inference_device": device, "new_interpreter": True, "bundle_only_hf_cache": True,
        "offline_mode": True, "cross_platform_validated": False, "performance_benchmark": False,
        **result,
    }


def bundle_readme(scope):
    role = (f"Formal {scope['run_completed_steps']:,}-step training run; this file may be an earlier best checkpoint."
            if scope["formal_training_run"] else "SMOKE / PARTIAL RUN ONLY — not the final trained policy.")
    return f"""# AIRBOT depth inference bundle

{role}

Selected checkpoint: step {scope['checkpoint_training_step']}. Completed training run:
{scope['run_completed_steps']} steps, {scope['train_episodes']} training episodes and
{scope['validation_episodes']} validation episodes. This package has only offline inference
validation; it does not establish real-robot grasp success or authorize robot motion.
The checkpoint originated in the {scope['checkpoint_origin_phase']} phase, whose saved
training target was {scope['checkpoint_origin_run_target_steps']} steps. Its actual step
and original config are preserved even when a continuation retains an earlier best.

`manifest.json` identifies the source checkpoint, stripped inference checkpoint, audit,
runtime files and cached model weights by SHA256. It hashes every packaged file except
itself, avoiding a self-referential checksum. `bundle-validation.json` records a fresh-process
offline comparison against the original checkpoint. `audit.json` preserves the training
audit; its historical absolute paths are provenance, not runtime inputs.

Use an already installed compatible Python environment. `runtime-versions.json` records
the verified dependency versions, CUDA and machine. No virtual environment is included.
`openpi_client` is an external dependency, not vendored here. Orin/other Python or CUDA
environments require a new offline comparison: strict preprocessing provenance rejects
different library versions or implementation bytes. This package does not claim cross-platform
compatibility, and no installer changes your environment.

From this directory, with your compatible interpreter selected:

```bash
export AIRBOT_DEPTH_PYTHON=/path/to/compatible/python
./serve.sh
```

The server defaults to CUDA and unauthenticated `127.0.0.1:8026`. Existing trusted SSH
port forwarding can connect a remote client. Arguments such as `--device cpu` or
`--port 8027` are forwarded; the checkpoint is fixed to this bundle's `policy.pt`.
No robot SDK is imported or hardware enabled. `/healthz` is a liveness check.

For offline RGB-to-action inference without starting a server, choose a new output filename:

```bash
env -u PYTHONPATH -u TRANSFORMERS_CACHE \\
  HF_HUB_CACHE="$PWD/hf_hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \\
  "$AIRBOT_DEPTH_PYTHON" -m airbot_depth.policy --checkpoint "$PWD/policy.pt" \\
  --observation "$PWD/sample_observation.npz" --output /tmp/airbot-depth-actions.json \\
  --device cuda --local-files-only
```

The policy consumes native absolute state `[joint1,...,joint6,gripper]`, RGB head/wrist
and its exact saved prompt; it returns absolute `[8,7]` action chunks in radians/metres.
Use the metadata for checkpoint-specific dimensions. For this paper-bag task the existing
Airbot client must explicitly use `--chunk-size-execute 4` (never above horizon 8) and
`--step-rate 25`. Mechanical limits, zeroing, camera setup and real-robot authorization
remain separate. The copied project notes are in `docs/airbot-paperbag-depth.md`.
"""


def export_bundle(checkpoint, audit_path, output, device="cuda", threads=4):
    checkpoint, audit_path = Path(checkpoint).resolve(strict=True), Path(audit_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {output}")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    checkpoint_sha256, audit_sha256 = sha256_file(checkpoint), sha256_file(audit_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint changed while being read")
    audit = json.loads(audit_path.read_text())
    scope = verify_audit_claims(payload, audit, checkpoint_sha256)
    sources = verify_runtime_sources(payload, audit)
    artifacts = resolve_hf_artifacts(payload)
    sample_record = audit["artifacts"]["sample_observation.npz"]
    sample = Path(sample_record["path"]).resolve(strict=True)
    if sample != Path(audit["sample_observation"]["path"]).resolve(strict=True):
        raise ValueError("Audit contains inconsistent sample-observation paths")
    if sha256_file(sample) != sample_record["sha256"]:
        raise ValueError("Audited sample observation changed")
    versions = runtime_versions(device)
    stripped = inference_payload(payload, checkpoint_sha256, scope)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", {"schema_version": 1, "status": "building", "scope": scope})
    try:
        for record in sources:
            destination = output / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / record["path"], destination)
            if sha256_file(destination) != record["sha256"]:
                raise ValueError(f"Runtime source changed while copying: {record['path']}")
        model = payload["depth_config"]
        snapshot = output / "hf_hub" / ("models--" + model["model_id"].replace("/", "--")) / "snapshots" / model["revision"]
        snapshot.mkdir(parents=True)
        for name, source in artifacts.items():
            shutil.copyfile(source, snapshot / name)
            if sha256_file(snapshot / name) != payload["depth_provenance"]["files_sha256"][name]:
                raise ValueError(f"Depth model artifact changed while copying: {name}")
        torch.save(stripped, output / "policy.pt")
        shutil.copyfile(audit_path, output / "audit.json")
        shutil.copyfile(sample, output / "sample_observation.npz")
        if (sha256_file(output / "audit.json") != audit_sha256
                or sha256_file(output / "sample_observation.npz") != sample_record["sha256"]):
            raise ValueError("Audit or observation changed while copying")
        (output / "docs").mkdir()
        shutil.copyfile(ROOT / "docs/airbot-paperbag-depth.md", output / "docs/airbot-paperbag-depth.md")
        shutil.copyfile(ROOT / "templates/airbot_depth_serve.sh", output / "serve.sh")
        (output / "serve.sh").chmod(0o755)
        (output / "README.md").write_text(bundle_readme(scope), encoding="utf-8")
        atomic_json(output / "runtime-versions.json", versions)
        print(json.dumps({"stage": "validating_isolated_bundle", "output": str(output), "scope": scope}), flush=True)
        validation = validate_bundle(output, checkpoint, device, threads)
        if validation["source_checkpoint_sha256"] != checkpoint_sha256 or sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("Source checkpoint changed during bundle validation")
        atomic_json(output / "bundle-validation.json", validation)
        files = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"]
        manifest = {
            "schema_version": 1, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope, "source_checkpoint_sha256": checkpoint_sha256,
            "exported_checkpoint_sha256": sha256_file(output / "policy.pt"),
            "audit_sha256": audit_sha256, "sample_observation_sha256": sample_record["sha256"],
            "runtime_source_verification": sources,
            "removed_checkpoint_fields": stripped["inference_export"]["training_state_removed"],
            "exporter_sha256": sha256_file(__file__), "files": files,
            "file_inventory_excludes": ["manifest.json"],
            "robot_executed": False, "cross_platform_validated": False,
        }
        atomic_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        atomic_json(output / "manifest.json", {"schema_version": 1, "status": "failed", "scope": scope,
                                              "error": f"{type(error).__name__}: {error}", "robot_executed": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted checkpoint explicitly covered by the audit")
    parser.add_argument("--audit", required=True, help="Passed airbot_depth.audit report.json")
    parser.add_argument("--output", required=True, help="New bundle directory; existing paths are refused")
    parser.add_argument("--device", default="cuda", help="Local device for original/bundle inference comparison")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    result = export_bundle(args.checkpoint, args.audit, args.output, device=args.device, threads=args.threads)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "scope": result["scope"], "files": len(result["files"]), "robot_executed": False}), flush=True)


if __name__ == "__main__":
    main()
