#!/usr/bin/env bash
set -euo pipefail

airbot_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for airbot_argument in "$@"; do
    case "$airbot_argument" in
        --checkpoint|--checkpoint=*)
            printf '%s\n' 'The checkpoint is fixed to this bundle/policy.pt.' >&2
            exit 2
            ;;
    esac
done
export HF_HUB_CACHE="$airbot_bundle_dir/hf_hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH TRANSFORMERS_CACHE
cd -- "$airbot_bundle_dir"
exec "${AIRBOT_DEPTH_PYTHON:-python3}" -m airbot_depth.serve \
    --device cuda --local-files-only --host 127.0.0.1 --port 8026 \
    "$@" --checkpoint "$airbot_bundle_dir/policy.pt"
