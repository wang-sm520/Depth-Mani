#!/usr/bin/env bash
# Build an official Torchvision tag around the existing, read-only Jetson Torch.
# All generated files stay inside this deployment; no shared environment is installed into.
set -euo pipefail

DEPLOY_ROOT=/mnt/nvme/pi05/airbot-paperbag200-da2-orin-20260924
SOURCE_PYTHON=/mnt/nvme/home/miniconda3/envs/anygrasp/bin/python
CUDA_BASE="$DEPLOY_ROOT/cuda-base"
EVIDENCE="$DEPLOY_ROOT/runtime/vision-bootstrap-v0181"
VISION_COMMIT=126fc22ce33e6c2426edcf9ed540810c178fe9ce
VISION_TAG=v0.18.1
EXPECTED_TORCH=2.4.0a0+07cecf4168.nv24.05
RESUME_CREATED_BASE=false
if [[ ${1:-} == --resume-created-base && $# -eq 1 ]]; then
    RESUME_CREATED_BASE=true
elif [[ $# -ne 0 ]]; then
    printf 'Only the explicit --resume-created-base recovery option is accepted.\n' >&2
    exit 2
fi

if [[ "$RESUME_CREATED_BASE" == true ]]; then
    [[ -f "$EVIDENCE/base-before.json" && -f "$CUDA_BASE/pyvenv.cfg" && ! -L "$CUDA_BASE" ]]
elif [[ -e "$CUDA_BASE" || -L "$CUDA_BASE" ]]; then
    printf 'Refusing existing CUDA base: %s\n' "$CUDA_BASE" >&2
    exit 2
fi
[[ $(uname -m) == aarch64 ]]
[[ -d "$DEPLOY_ROOT" && ! -L "$DEPLOY_ROOT" ]]
[[ -f "$DEPLOY_ROOT/prepare_env.py" && -f "$DEPLOY_ROOT/model/runtime-versions.json" ]]
[[ -x "$SOURCE_PYTHON" && -x /usr/local/cuda-12.2/bin/nvcc ]]
[[ ! -e "$EVIDENCE/source" ]]
if [[ "$RESUME_CREATED_BASE" == false ]]; then
    [[ ! -e "$EVIDENCE/base-before.json" ]]
fi
mkdir -p "$EVIDENCE/tmp" "$EVIDENCE/cache" "$EVIDENCE/wheels"

unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$EVIDENCE/tmp" XDG_CACHE_HOME="$EVIDENCE/cache"
export TORCH_EXTENSIONS_DIR="$EVIDENCE/torch-extensions"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
export CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.7
export CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export FORCE_CUDA=1 TORCHVISION_USE_FFMPEG=0 TORCHVISION_USE_VIDEO_CODEC=0
export TORCHVISION_USE_NVJPEG=0 NVCC_FLAGS='--threads=1'
export PATH="$CUDA_BASE/bin:/usr/local/cuda-12.2/bin:/usr/bin:/bin"

printf 'Checking the inherited CUDA Torch without modifying it.\n'
if [[ "$RESUME_CREATED_BASE" == false ]]; then
    "$SOURCE_PYTHON" -I -B "$DEPLOY_ROOT/prepare_env.py" inspect \
        --runtime-versions "$DEPLOY_ROOT/model/runtime-versions.json" \
        --python "$SOURCE_PYTHON" --probe-timeout 60 \
        --report "$EVIDENCE/base-before.json" > "$EVIDENCE/base-before.console.log"
fi
"$SOURCE_PYTHON" -I -B - "$EVIDENCE/base-before.json" "$EXPECTED_TORCH" <<'PY'
import json, sys, torch
import torch._custom_ops
probe = json.load(open(sys.argv[1]))['python_candidates'][0]
assert str(torch.__version__) == sys.argv[2], probe
assert probe['machine'] == 'aarch64' and probe['cuda_test_passed'], probe
assert probe['cuda_runtime'] == '12.2' and probe['cudnn_version'] == 8904, probe
assert not hasattr(torch.library, 'register_fake') and hasattr(torch.library, 'impl_abstract')
assert hasattr(torch._custom_ops, 'impl_abstract')
assert probe['packages']['torchvision'] is None, probe
PY

if [[ "$RESUME_CREATED_BASE" == false ]]; then
    "$SOURCE_PYTHON" -I -B -m venv --system-site-packages "$CUDA_BASE"
fi
"$CUDA_BASE/bin/python" -I -B - "$CUDA_BASE" "$EVIDENCE/base-before.json" <<'PY'
import json, pathlib, sys, sysconfig, torch
target = pathlib.Path(sys.argv[1]).resolve()
probe = json.load(open(sys.argv[2]))['python_candidates'][0]
assert pathlib.Path(sys.prefix).resolve() == target
assert pathlib.Path(sysconfig.get_path('purelib')).resolve().is_relative_to(target)
assert str(pathlib.Path(torch.__file__).resolve()) == probe['torch_identity']['module_file']
assert str(torch.__version__) == probe['torch_identity']['version']
assert torch.cuda.is_available()
print('Read-only CUDA Torch inheritance verified:', torch.__file__)
PY

printf 'Installing build tools only into the new CUDA base.\n'
if ! "$CUDA_BASE/bin/python" -I -B - "$CUDA_BASE" <<'PY'
import importlib.metadata as md, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
for name, expected in {'setuptools': '69.5.1', 'wheel': '0.45.1', 'ninja': '1.11.1.4'}.items():
    local = [distribution for distribution in md.distributions()
             if distribution.metadata.get('Name', '').lower() == name
             and pathlib.Path(distribution.locate_file('')).resolve().is_relative_to(root)]
    if len(local) != 1 or local[0].version != expected or md.version(name) != expected:
        raise SystemExit(1)
print('The three pinned build tools are already present only in the new venv.')
PY
then
"$CUDA_BASE/bin/python" -I -B -m pip --isolated install \
    --index-url https://pypi.org/simple --only-binary=:all: --no-deps \
    --force-reinstall --no-cache-dir --disable-pip-version-check \
    'setuptools==69.5.1' 'wheel==0.45.1' 'ninja==1.11.1.4' \
    2>&1 | tee "$EVIDENCE/build-tools.log"
fi

printf 'Fetching unmodified official Torchvision %s.\n' "$VISION_TAG"
if [[ -f "$EVIDENCE/official-source-snapshot.tar.gz" ]]; then
    (
        cd "$EVIDENCE"
        sha256sum --check official-source-snapshot.sha256
        tar --extract --gzip --no-same-owner --file official-source-snapshot.tar.gz
    ) 2>&1 | tee "$EVIDENCE/source-fetch.log"
else
git -c http.version=HTTP/1.1 clone --depth 1 --branch "$VISION_TAG" https://github.com/pytorch/vision.git "$EVIDENCE/source" \
    2>&1 | tee "$EVIDENCE/source-fetch.log"
fi
[[ $(git -C "$EVIDENCE/source" rev-parse HEAD) == "$VISION_COMMIT" ]]
[[ $(git -C "$EVIDENCE/source" remote get-url origin) == https://github.com/pytorch/vision.git ]]
git -C "$EVIDENCE/source" diff --exit-code
export PYTORCH_VERSION="$EXPECTED_TORCH"
"$CUDA_BASE/bin/python" -I -B - "$EVIDENCE" "$SOURCE_PYTHON" "$CUDA_BASE" "$VISION_TAG" "$VISION_COMMIT" <<'PY'
import hashlib, json, os, pathlib, platform, subprocess, sys, torch
root = pathlib.Path(sys.argv[1])
record = {
    'schema_version': 1, 'source_repository': 'https://github.com/pytorch/vision.git',
    'source_tag': sys.argv[4], 'source_commit': sys.argv[5],
    'source_python': sys.argv[2], 'cuda_base': sys.argv[3], 'python': sys.version,
    'torch': str(torch.__version__), 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(), 'machine': platform.machine(),
    'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'environment': {key: os.environ[key] for key in (
        'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL',
        'PYTORCH_VERSION', 'FORCE_CUDA', 'NVCC_FLAGS', 'CC', 'CXX',
        'TORCHVISION_USE_FFMPEG', 'TORCHVISION_USE_VIDEO_CODEC', 'TORCHVISION_USE_NVJPEG')},
    'compatibility_selection': 'Torch has impl_abstract but lacks register_fake; official vision v0.18.1 uses impl_abstract',
    'bootstrap_sha256': hashlib.sha256((root / 'bootstrap_airbot_orin_vision.sh').read_bytes()).hexdigest(),
    'nvcc': subprocess.check_output(['/usr/local/cuda-12.2/bin/nvcc', '--version'], text=True),
}
with (root / 'build-inputs.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
PY

printf 'Building CUDA extension for Orin sm_87 with at most four compiler jobs.\n'
(
    cd "$EVIDENCE/source"
    "$CUDA_BASE/bin/python" -B -m pip --isolated wheel . --no-deps --no-build-isolation \
        --no-cache-dir --disable-pip-version-check --wheel-dir "$EVIDENCE/wheels" --verbose
) 2>&1 | tee "$EVIDENCE/torchvision-build.log"
git -C "$EVIDENCE/source" diff --exit-code
shopt -s nullglob
VISION_WHEELS=("$EVIDENCE"/wheels/torchvision-*.whl)
[[ ${#VISION_WHEELS[@]} -eq 1 ]]
sha256sum "${VISION_WHEELS[0]}" | tee "$EVIDENCE/wheel.sha256"
"$CUDA_BASE/bin/python" -I -B - "${VISION_WHEELS[0]}" "$EXPECTED_TORCH" <<'PY'
import email, sys, zipfile
from packaging.requirements import Requirement
from packaging.version import Version
with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata = email.message_from_bytes(wheel.read(next(name for name in wheel.namelist() if name.endswith('.dist-info/METADATA'))))
requirements = metadata.get_all('Requires-Dist') or []
torch_requirements = [Requirement(line) for line in requirements if Requirement(line).name == 'torch']
assert len(torch_requirements) == 1, requirements
specifiers = list(torch_requirements[0].specifier)
assert len(specifiers) == 1 and specifiers[0].operator == '==' and Version(specifiers[0].version) == Version(sys.argv[2]), requirements
print('Wheel dependencies:', requirements)
PY
"$CUDA_BASE/bin/python" -I -B -m pip --isolated install --no-deps --no-cache-dir \
    --disable-pip-version-check "${VISION_WHEELS[0]}" 2>&1 | tee "$EVIDENCE/torchvision-install.log"

printf 'Checking clean imports, CUDA matrix/convolution and compiled Torchvision NMS.\n'
"$CUDA_BASE/bin/python" -I -B - "$EVIDENCE/cuda-verification.json" "$EXPECTED_TORCH" <<'PY'
import importlib.metadata as md, json, pathlib, sys
import torch, torchvision
assert str(torch.__version__) == sys.argv[2]
assert torch.cuda.is_available() and torchvision.extension._has_ops()
with torch.inference_mode():
    values = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
    product = values @ values.T
    image = torch.ones((1, 3, 8, 8), device='cuda')
    kernel = torch.ones((4, 3, 3, 3), device='cuda')
    convolution = torch.nn.functional.conv2d(image, kernel)
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.], [5., 5., 6., 6.]], device='cuda')
    scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    assert torch.equal(convolution.cpu(), torch.full((1, 4, 6, 6), 27.))
    assert torch.allclose(product.cpu(), values.cpu() @ values.cpu().T)
    assert keep.cpu().tolist() == [0, 2]
record = {
    'schema_version': 1, 'status': 'passed', 'python': sys.version,
    'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
    'torch_file': torch.__file__, 'torchvision_file': torchvision.__file__,
    'torchvision_git_version': torchvision.version.git_version,
    'torchvision_requires': md.requires('torchvision'),
    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
    'gpu': torch.cuda.get_device_name(0), 'cuda_architectures': torch.cuda.get_arch_list(),
    'matrix_sum': product.sum().item(), 'convolution_sum': convolution.sum().item(),
    'torchvision_nms_indices': keep.cpu().tolist(), 'model_validated': False, 'robot_executed': False,
}
with pathlib.Path(sys.argv[1]).open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record, indent=2))
PY

"$CUDA_BASE/bin/python" -I -B "$DEPLOY_ROOT/prepare_env.py" inspect \
    --runtime-versions "$DEPLOY_ROOT/model/runtime-versions.json" \
    --python "$CUDA_BASE/bin/python" --probe-timeout 60 \
    --report "$EVIDENCE/cuda-base-inspect.json" > "$EVIDENCE/cuda-base-inspect.console.log"
"$SOURCE_PYTHON" -I -B "$DEPLOY_ROOT/prepare_env.py" inspect \
    --runtime-versions "$DEPLOY_ROOT/model/runtime-versions.json" \
    --python "$SOURCE_PYTHON" --probe-timeout 60 \
    --report "$EVIDENCE/base-after.json" > "$EVIDENCE/base-after.console.log"
"$CUDA_BASE/bin/python" -I -B - "$EVIDENCE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
before = json.loads((root / 'base-before.json').read_text())['python_candidates'][0]
after = json.loads((root / 'base-after.json').read_text())['python_candidates'][0]
prepared = json.loads((root / 'cuda-base-inspect.json').read_text())['python_candidates'][0]
for key in ('torch_identity', 'protected_distributions', 'packages', 'cuda_runtime', 'cudnn_version'):
    assert before[key] == after[key], ('Shared base changed', key)
assert prepared['eligible_cuda_base'], prepared
assert before['torch_identity'] == prepared['torch_identity']
record = {'schema_version': 1, 'status': 'eligible_cuda_base',
          'source_environment_unchanged': True, 'source_torch_hashes_unchanged': True,
          'eligible_cuda_base': True, 'inference_ready': False,
          'cross_platform_validated': False, 'robot_executed': False,
          'python': prepared['requested_python'], 'torch_identity': prepared['torch_identity'],
          'torchvision_identity': prepared['torchvision_identity']}
with (root / 'result.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
files = [path for path in root.iterdir() if path.is_file() and path.name != 'SHA256SUMS' and not path.name.startswith('launcher')]
files.extend(sorted((root / 'wheels').glob('*.whl')))
with (root / 'SHA256SUMS').open('x') as stream:
    for path in sorted(files):
        stream.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n')
print(json.dumps(record, indent=2))
PY
printf 'CUDA base ready for the separate model environment setup.\n'
