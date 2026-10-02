#!/usr/bin/env python3
"""Inspect an Orin host, or prepare a new NVMe inference venv without replacing CUDA Torch.

Only official PyPI binary wheels for the recorded non-CUDA packages are installed,
with dependency resolution disabled. Successful setup still requires the separate
cross-platform inference attestation. No robot SDK is imported or enabled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
NVME_ROOT = Path("/mnt/nvme")
DEFAULT_VERSIONS = ROOT / "deploy/airbot-paperbag200-da2-100k-20260924/runtime-versions.json"
MANAGED_PACKAGES = ("numpy", "transformers", "pillow", "huggingface-hub", "h5py", "safetensors",
                    "tokenizers", "msgpack", "websockets", "typing-extensions")
PROTECTED_PACKAGES = ("torch", "torchvision", "torchaudio")
CANDIDATE_PYTHONS = (
    "/usr/bin/python3",
    "/mnt/nvme/pi05/runtime/official-20260910/openpi_v0.2.0/.venv/bin/python",
    "/mnt/nvme/pi05/runtime/official-20260910/robot-venv/bin/python",
    "/mnt/nvme/pi05/official-20260910/openpi_v0.2.0/.venv/bin/python",
    "/mnt/nvme/pi05/official-20260910/robot-venv/bin/python",
)


PROBE_CODE = r'''
import hashlib, importlib, importlib.metadata as md, json, pathlib, platform, site, sys

def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def identity(module, names):
    root = pathlib.Path(module.__file__).resolve().parent
    paths = [pathlib.Path(module.__file__).resolve()]
    for pattern in names:
        paths.extend(sorted(root.glob(pattern)))
    return {'version': str(module.__version__), 'module_file': str(pathlib.Path(module.__file__).resolve()),
            'files_sha256': {str(path): digest(path) for path in paths if path.is_file()}}

result = {'status': 'probed', 'python': sys.version, 'executable': sys.executable,
          'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'machine': platform.machine(),
          'site_dirs': [str(pathlib.Path(path).resolve()) for path in site.getsitepackages()
                        if pathlib.Path(path).is_dir()], 'packages': {}, 'protected_distributions': {}, 'errors': []}
for name in ('numpy','torch','torchvision','torchaudio','transformers','pillow','huggingface-hub',
             'h5py','safetensors','tokenizers','openpi-client','msgpack','websockets','typing-extensions'):
    try:
        result['packages'][name] = md.version(name)
    except md.PackageNotFoundError:
        result['packages'][name] = None
for distribution in md.distributions():
    name = (distribution.metadata.get('Name') or '').lower().replace('_','-')
    if name in {'torch','torchvision','torchaudio','triton','cuda-python','tensorrt'} or name.startswith(('nvidia-','pytorch-triton')):
        result['protected_distributions'].setdefault(name, distribution.version)
try:
    import torch
    result['torch_identity'] = identity(torch, ('_C*.so','lib/libtorch_cuda.so','lib/libc10_cuda.so'))
    result['cuda_available'] = bool(torch.cuda.is_available())
    result['cuda_runtime'] = torch.version.cuda
    result['cudnn_version'] = torch.backends.cudnn.version()
    result['cuda_test_passed'] = False
    if result['cuda_available']:
        with torch.inference_mode():
            values = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4,4)
            product = values @ values.T
            torch.cuda.synchronize()
            result['cuda_test_passed'] = bool(torch.isfinite(product).all().item())
            result['cuda_test_sum'] = float(product.sum().item())
        result['gpu'] = torch.cuda.get_device_name(0)
        result['cuda_device_count'] = torch.cuda.device_count()
except Exception as error:
    result['errors'].append('torch: ' + type(error).__name__ + ': ' + str(error)[:2000])
try:
    import torchvision
    result['torchvision_identity'] = identity(torchvision, ('_C*.so','image*.so'))
except Exception as error:
    result['errors'].append('torchvision: ' + type(error).__name__ + ': ' + str(error)[:2000])
result['eligible_cuda_base'] = bool(result.get('cuda_test_passed') and result.get('torchvision_identity') and not result['errors'])
print('AIRBOT_ENV_PROBE=' + json.dumps(result, allow_nan=False))
'''


VERIFY_CODE = r'''
import importlib, importlib.metadata as md, json, pathlib, sys
expected = json.loads(sys.argv[1])
vendor = sys.argv[2]
result = {'version_errors': [], 'import_errors': [], 'dependency_errors': [], 'module_files': {}}
for name, wanted in expected.items():
    try:
        actual = md.version(name)
        if actual != wanted:
            result['version_errors'].append({'package': name, 'expected': wanted, 'actual': actual})
    except md.PackageNotFoundError:
        result['version_errors'].append({'package': name, 'expected': wanted, 'actual': None})
modules = ['numpy','PIL','transformers','huggingface_hub','h5py','safetensors','tokenizers',
           'msgpack','websockets.sync.client','typing_extensions']
if vendor:
    modules.append('openpi_client.msgpack_numpy')
for name in modules:
    try:
        module = importlib.import_module(name)
        result['module_files'][name] = str(pathlib.Path(module.__file__).resolve())
    except Exception as error:
        result['import_errors'].append(name + ': ' + type(error).__name__ + ': ' + str(error)[:1500])
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTImageProcessor, DepthAnythingForDepthEstimation
    from transformers.utils import is_torch_available
    if not is_torch_available():
        raise RuntimeError('Transformers does not recognize this installed Torch version as supported')
    import numpy as np, torch
    values = torch.from_numpy(np.ones((2,2), dtype=np.float32)).to('cuda')
    torch.cuda.synchronize()
    assert float(values.sum().cpu().item()) == 4.0
    result['depth_api_and_numpy_cuda_interop'] = True
except Exception as error:
    result['import_errors'].append('depth API/NumPy CUDA interop: ' + type(error).__name__ + ': ' + str(error)[:1500])
try:
    from packaging.requirements import Requirement
    checked, pending = set(), list(expected) + ['torch','torchvision']
    while pending:
        name = pending.pop()
        normalized = name.lower().replace('_','-')
        if normalized in checked:
            continue
        checked.add(normalized)
        try:
            requirements = md.requires(name) or []
        except md.PackageNotFoundError:
            result['dependency_errors'].append({'package': name, 'error': 'not installed'})
            continue
        for text in requirements:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
                continue
            try:
                actual = md.version(requirement.name)
            except md.PackageNotFoundError:
                actual = None
            if actual is None or (requirement.specifier and not requirement.specifier.contains(actual, prereleases=True)):
                result['dependency_errors'].append({'required_by': name, 'requirement': text, 'installed': actual})
            elif requirement.name.lower().replace('_','-') not in checked:
                pending.append(requirement.name)
    result['dependency_packages_checked'] = sorted(checked)
except Exception as error:
    result['dependency_errors'].append({'error': type(error).__name__ + ': ' + str(error)[:1500]})
if vendor:
    actual = result['module_files'].get('openpi_client.msgpack_numpy')
    expected_path = pathlib.Path(vendor).resolve() / 'openpi_client/msgpack_numpy.py'
    if actual != str(expected_path):
        result['import_errors'].append('OpenPI codec did not load from the explicitly supplied local snapshot')
result['passed'] = not any(result[name] for name in ('version_errors','import_errors','dependency_errors'))
print('AIRBOT_ENV_VERIFY=' + json.dumps(result, allow_nan=False))
'''


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def subprocess_environment():
    result = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "XLA_FLAGS", "JAX_PLATFORMS"):
        result.pop(name, None)
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE=os.devnull)
    return result


def run_readonly(command, timeout=30):
    try:
        result = subprocess.run([str(value) for value in command], env=subprocess_environment(),
                                capture_output=True, text=True, timeout=timeout)
        return {"command": [str(value) for value in command], "returncode": result.returncode,
                "stdout": result.stdout[-20000:], "stderr": result.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": [str(value) for value in command], "returncode": None,
                "stdout": "", "stderr": f"{type(error).__name__}: {error}"[:4000]}


def parse_marked_json(result, marker):
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or f"Command failed with exit code {result['returncode']}")
    for line in reversed(result["stdout"].splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise ValueError("Python probe did not emit its structured result")


def probe_python(path, timeout=30):
    # Preserve the venv launcher path: resolving its symlink would select the
    # underlying system Python and silently inspect a different environment.
    executable = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return {"status": "missing", "requested_python": str(executable), "eligible_cuda_base": False}
    result = run_readonly([executable, "-I", "-B", "-c", PROBE_CODE], timeout)
    try:
        report = parse_marked_json(result, "AIRBOT_ENV_PROBE=")
        report["requested_python"] = str(executable)
        return report
    except (ValueError, RuntimeError) as error:
        return {"status": "failed", "requested_python": str(executable), "eligible_cuda_base": False,
                "error": str(error), "stderr": result["stderr"]}


def read_optional(path):
    try:
        return Path(path).read_text(errors="replace").strip().replace("\x00", "")
    except OSError:
        return None


def mount_record(target):
    result = run_readonly(["findmnt", "--json", "--target", target,
                           "--output", "TARGET,SOURCE,FSTYPE,MAJ:MIN,OPTIONS"], 10)
    if result["returncode"] != 0:
        return {"error": result["stderr"] or f"findmnt could not resolve {target} (exit {result['returncode']})",
                "target": str(target)}
    try:
        record = json.loads(result["stdout"])["filesystems"][0]
        return {"target": record["target"], "source": record["source"], "fstype": record["fstype"],
                "major_minor": record["maj:min"], "options": record["options"].split(",")}
    except (ValueError, KeyError, IndexError) as error:
        return {"target": str(target), "error": str(error)}


def inspect_mount():
    mounted, root = mount_record(str(NVME_ROOT)), mount_record("/")
    result = {"requested_mountpoint": str(NVME_ROOT), "mount": mounted, "root_mount": root,
              "is_mountpoint": False, "is_nvme_block_device": False, "is_independent_device": False}
    result["df"] = run_readonly(["df", "-B1", "--output=size,used,avail,target", str(NVME_ROOT)], 10)
    if "error" in mounted or "error" in root:
        return result
    result["is_mountpoint"] = (mounted["target"] == str(NVME_ROOT) and NVME_ROOT.exists()
                                and NVME_ROOT.resolve() == NVME_ROOT and not NVME_ROOT.is_symlink())
    source = Path(mounted["source"].split("[", 1)[0])
    try:
        source = source.resolve(strict=True)
        source_stat = source.stat()
        major_minor = f"{os.major(source_stat.st_rdev)}:{os.minor(source_stat.st_rdev)}"
        sysfs = (Path("/sys/dev/block") / major_minor).resolve(strict=True)
        result.update(source_realpath=str(source), sysfs_path=str(sysfs))
        result["is_nvme_block_device"] = (stat.S_ISBLK(source_stat.st_mode) and major_minor == mounted["major_minor"]
                                           and any(re.fullmatch(r"nvme\d+n\d+(p\d+)?", part) for part in sysfs.parts))
    except OSError as error:
        result["device_error"] = str(error)
    result["is_independent_device"] = mounted["major_minor"] != root["major_minor"]
    try:
        result["free_bytes"] = shutil.disk_usage(NVME_ROOT).free
    except OSError:
        result["free_bytes"] = None
    return result


def load_versions(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("packages"), dict):
        raise ValueError("Expected the exported schema-1 runtime-versions.json")
    for name in (*MANAGED_PACKAGES, "torch", "torchvision", "openpi-client"):
        version = data["packages"].get(name)
        if not isinstance(version, str) or re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version) is None:
            raise ValueError(f"Missing or unsupported exported package version: {name}")
    return {"path": str(path), "sha256": sha256_file(path), "recorded_runtime": data,
            "managed_packages": {name: data["packages"][name] for name in MANAGED_PACKAGES}}


def inspect_host(versions, candidates, timeout):
    return {"schema_version": 1, "operation": "inspect", "readonly": True, "robot_enabled": False,
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "platform": platform.platform(), "l4t_release": read_optional("/etc/nv_tegra_release"),
                         "board_model": read_optional("/proc/device-tree/model"),
                         "cuda_version_json": read_optional("/usr/local/cuda/version.json"),
                         "cuda_version_txt": read_optional("/usr/local/cuda/version.txt")},
            "mount": inspect_mount(), "versions": versions,
            "python_candidates": [probe_python(path, timeout) for path in dict.fromkeys(candidates)],
            "inference_ready": False, "cross_platform_attestation_required": True}


def validate_create_host(host, output):
    if host["platform"]["system"] != "Linux" or host["platform"]["machine"] != "aarch64":
        raise ValueError("Environment creation is permitted only on the real aarch64 Linux Orin host")
    if not host["platform"].get("l4t_release") or "orin" not in (host["platform"].get("board_model") or "").lower():
        raise ValueError("An NVIDIA L4T release and Orin board identity must be detected")
    mount = host["mount"]
    if not all(mount.get(key) is True for key in ("is_mountpoint", "is_nvme_block_device")):
        raise ValueError("/mnt/nvme must be an actual mount backed by a verified NVMe block device, not an ordinary directory or symlink")
    if "rw" not in mount["mount"].get("options", []):
        raise ValueError("NVMe mount is not writable")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output must be a new environment path; existing or partial outputs are never overwritten")
    if not output.resolve().is_relative_to(NVME_ROOT) or output.resolve() == NVME_ROOT:
        raise ValueError("The entire new environment must be below the actual /mnt/nvme mount")
    if mount.get("free_bytes") is None or mount["free_bytes"] < 2 * 1024**3:
        raise ValueError("At least 2 GiB free on the NVMe mount is required for the isolated dependency layer")


def verify_protected_identity(base, current):
    for key in ("torch_identity", "torchvision_identity", "protected_distributions", "cuda_runtime", "cudnn_version"):
        if not base.get(key) or current.get(key) != base[key]:
            raise ValueError(f"Base CUDA runtime identity changed or was shadowed: {key}")
    if not current.get("eligible_cuda_base"):
        raise ValueError("The selected Torch/Torchvision pair no longer passes CUDA inference probing")


def constraints_text(base, packages):
    if set(packages) != set(MANAGED_PACKAGES):
        raise ValueError("Only the explicit non-CUDA inference package allowlist may be installed")
    protected = base.get("protected_distributions", {})
    if any(name not in protected for name in ("torch", "torchvision")):
        raise ValueError("The base must expose installed Torch and Torchvision distribution metadata")
    for name, version in protected.items():
        if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version) is None:
            raise ValueError("Invalid protected distribution identity")
    return "# CUDA packages below are constraints only: never installation targets.\n" + "".join(
        f"{name}=={version}\n" for name, version in sorted({**protected, **packages}.items()))


def install_command(python, constraints, packages):
    if set(packages) != set(MANAGED_PACKAGES):
        raise ValueError("Refusing unknown/CUDA package installation targets")
    return [str(python), "-I", "-B", "-m", "pip", "--isolated", "install", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--no-deps", "--ignore-installed", "--no-cache-dir", "--disable-pip-version-check",
            "--timeout", "15", "--retries", "1", "--constraint", str(constraints),
            *[f"{name}=={packages[name]}" for name in MANAGED_PACKAGES]]


def write_new_text(path, content):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)


def create_environment(args, versions):
    output = Path(os.path.abspath(os.path.expanduser(str(args.output))))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists; partial and completed environments are preserved")
    host = inspect_host(versions, [], args.probe_timeout)
    validate_create_host(host, output)
    base_python = Path(os.path.abspath(os.path.expanduser(str(args.base_python))))
    base = probe_python(base_python, args.probe_timeout)
    if not base.get("eligible_cuda_base") or base.get("machine") != "aarch64":
        raise ValueError("Base Python must already import Torch/Torchvision and pass a real CUDA tensor operation")
    packages = versions["managed_packages"]
    constraints = constraints_text(base, packages)
    vendor = None
    if args.openpi_client_root is not None:
        vendor = Path(args.openpi_client_root).resolve(strict=True)
        if not (vendor / "openpi_client/__init__.py").is_file() or not (vendor / "openpi_client/msgpack_numpy.py").is_file():
            raise ValueError("--openpi-client-root must contain the trusted local openpi_client package")
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "setup-state.json"
    state = {"schema_version": 1, "status": "partial", "stage": "preflight_passed",
             "created_at": datetime.now(timezone.utc).isoformat(), "output": str(output),
             "base_python": str(base_python), "host": host, "base_probe": base, "versions": versions,
             "inference_ready": False, "cross_platform_attestation_required": True, "robot_enabled": False,
             "vendored_openpi_client_required": vendor is None, "commands": [],
             "driver_sha256": sha256_file(__file__)}
    atomic_json(state_path, state)
    try:
        command = [str(base_python), "-I", "-B", "-m", "venv", "--system-site-packages", str(output)]
        state["commands"].append(command)
        result = run_readonly(command, args.install_timeout)
        if result["returncode"] != 0:
            raise RuntimeError("venv creation failed: " + result["stderr"])
        python = output / "bin/python"
        result = run_readonly([python, "-I", "-B", "-c",
                               "import json,sys,sysconfig; print('AIRBOT_SITE='+json.dumps({'prefix':sys.prefix,'site':sysconfig.get_path('purelib')}))"],
                              args.probe_timeout)
        locations = parse_marked_json(result, "AIRBOT_SITE=")
        site_dir = Path(locations["site"]).resolve()
        if Path(locations["prefix"]).resolve() != output.resolve() or not site_dir.is_relative_to(output.resolve()):
            raise ValueError("New interpreter did not select the isolated output prefix/site-packages")
        site_dirs = [str(Path(path).resolve(strict=True)) for path in base["site_dirs"]]
        if not site_dirs:
            raise ValueError("Base interpreter reported no verified site-packages paths")
        inheritance = site_dir / "_airbot_verified_base.pth"
        write_new_text(inheritance, "import site; " + "; ".join(f"site.addsitedir({path!r})" for path in site_dirs) + "\n")
        state["base_inheritance"] = {"site_dirs": site_dirs, "path": str(inheritance), "sha256": sha256_file(inheritance),
                                     "mode": "read-only base site.addsitedir; new venv packages take precedence"}
        if vendor is not None:
            vendor_pth = site_dir / "_airbot_verified_openpi_client.pth"
            write_new_text(vendor_pth, f"import sys; sys.path.insert(0, {str(vendor)!r})\n")
            state["openpi_client_snapshot"] = {"root": str(vendor), "pth": str(vendor_pth),
                                               "files_sha256": {str(path.relative_to(vendor)): sha256_file(path)
                                                                for path in sorted((vendor / "openpi_client").rglob("*.py"))}}
        inherited = probe_python(python, args.probe_timeout)
        verify_protected_identity(base, inherited)
        state.update(stage="base_inheritance_verified", inherited_probe=inherited)
        atomic_json(state_path, state)
        constraints_path = output / "inference-constraints.txt"
        write_new_text(constraints_path, constraints)
        command = install_command(python, constraints_path, packages)
        state["commands"].append(command)
        state.update(stage="installing_pinned_non_cuda_packages", install_log=str(output / "install.log"))
        atomic_json(state_path, state)
        with (output / "install.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, env=subprocess_environment(), stdout=log, stderr=subprocess.STDOUT,
                                       text=True, timeout=args.install_timeout)
        if completed.returncode:
            raise RuntimeError(f"Pinned official-PyPI wheel installation failed (exit {completed.returncode}); see install.log")
        new_probe = probe_python(python, args.probe_timeout)
        base_after = probe_python(base_python, args.probe_timeout)
        verify_protected_identity(base, new_probe)
        verify_protected_identity(base, base_after)
        if base_after["packages"] != base["packages"]:
            raise ValueError("The original base environment package versions changed")
        state.update(stage="verifying_dependency_layer", new_probe=new_probe, base_after_probe=base_after,
                     base_environment_unchanged=True, protected_cuda_packages_unchanged=True)
        atomic_json(state_path, state)
        result = run_readonly([python, "-I", "-B", "-c", VERIFY_CODE, json.dumps(packages), str(vendor) if vendor else ""],
                              args.probe_timeout)
        verification = parse_marked_json(result, "AIRBOT_ENV_VERIFY=")
        state["dependency_verification"] = verification
        if not verification["passed"]:
            raise RuntimeError("The isolated layer has missing/conflicting dependencies or failed imports; see setup-state.json. "
                               "No transitive package or CUDA component was installed implicitly.")
        state.update(status="prepared_pending_attestation", stage="dependency_layer_verified",
                     completed_at=datetime.now(timezone.utc).isoformat(), inference_python=str(python),
                     pinned_packages_verified=True,
                     cuda_version_differences={name: {"exported": versions["recorded_runtime"]["packages"][name],
                                                      "orin_base": base["packages"][name]}
                                               for name in ("torch", "torchvision")
                                               if base["packages"][name] != versions["recorded_runtime"]["packages"][name]})
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state.update(status="partial", inference_ready=False, error={"type": type(error).__name__, "message": str(error)},
                     stopped_at=datetime.now(timezone.utc).isoformat())
        if not isinstance(error, (KeyboardInterrupt, SystemExit)):
            try:
                base_after = probe_python(base_python, args.probe_timeout)
                state["base_after_probe"] = base_after
                verify_protected_identity(base, base_after)
                if base_after["packages"] != base["packages"]:
                    raise ValueError("Base package versions changed during the failed preparation")
                state["base_environment_unchanged"] = True
            except Exception as verification_error:
                state["base_environment_unchanged"] = False
                state["base_verification_error"] = str(verification_error)
        atomic_json(state_path, state)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--runtime-versions", type=Path, default=DEFAULT_VERSIONS)
    shared.add_argument("--probe-timeout", type=float, default=30.0)
    inspect = commands.add_parser("inspect", parents=[shared], help="Read-only host/Python/CUDA/NVMe inventory")
    inspect.add_argument("--python", type=Path, action="append", help="Candidate interpreter; repeat to inspect several")
    inspect.add_argument("--report", type=Path, help="Optional new JSON inventory report")
    create = commands.add_parser("create", parents=[shared], help="Create a new NVMe dependency layer around verified installed Jetson Torch")
    create.add_argument("--base-python", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--openpi-client-root", type=Path, help="Trusted local directory containing openpi_client/; never fetched from PyPI")
    create.add_argument("--install-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if not 0 < args.probe_timeout <= 60 or (args.command == "create" and not 0 < args.install_timeout <= 1800):
        parser.error("probe-timeout must be in (0,60]; install-timeout must be in (0,1800]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "inspect" and args.report is not None and args.report.exists():
        raise FileExistsError("Inventory report already exists")
    versions = load_versions(args.runtime_versions)
    if args.command == "inspect":
        candidates = args.python or [sys.executable, *CANDIDATE_PYTHONS]
        report = inspect_host(versions, [str(path) for path in candidates], args.probe_timeout)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.report, report)
    else:
        report = create_environment(args, versions)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Orin environment preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
