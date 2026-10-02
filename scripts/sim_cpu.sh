#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSMESA="$ROOT/runtime/osmesa/usr/lib/x86_64-linux-gnu"
if [[ ! -e "$OSMESA/libOSMesa.so.8" ]]; then
  echo "Missing project-local OSMesa. See reports/progress.md for setup." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_DISABLE_JIT=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export LP_NUM_THREADS=4
export LD_LIBRARY_PATH="$OSMESA${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$@"
