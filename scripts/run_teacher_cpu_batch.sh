#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ATTEMPTS="${1:-1}"
SPLIT="${2:-train}"
LABEL="${3:-cpu-batch}"
TARGET_SUCCESSES="${4:-}"
CONFIG="${5:-configs/pilot.json}"
if [[ ! "$ATTEMPTS" =~ ^[1-9][0-9]*$ || ! "$LABEL" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: bash scripts/run_teacher_cpu_batch.sh ATTEMPTS SPLIT LABEL [TARGET_SUCCESSES] [CONFIG]" >&2
  exit 2
fi
case "$SPLIT" in train|validation|test) ;; *) exit 2 ;; esac
EXTRA_ARGS=()
if [[ -n "$TARGET_SUCCESSES" ]]; then
  if [[ ! "$TARGET_SUCCESSES" =~ ^[1-9][0-9]*$ ]]; then exit 2; fi
  EXTRA_ARGS=(--target-successes "$TARGET_SUCCESSES")
fi
mkdir -p runtime/logs
if [[ -e "runtime/logs/${LABEL}-server.log" || -e "runtime/logs/${LABEL}-collect.log" ]]; then
  echo "Choose a new label; existing batch logs will not be overwritten." >&2
  exit 1
fi
PORT="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_port"])' "$CONFIG")"
if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  echo "Teacher port already in use; refusing to replace its owner." >&2
  exit 1
fi
OPENPI_ROOT="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["openpi_root"])' "$CONFIG")"
TEACHER_PYTHON="$OPENPI_ROOT/.venv/bin/python"
CHECKPOINT="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_checkpoint"])' "$CONFIG")"
DATA_DIR="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_dir"])' "$CONFIG")"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4' \
  timeout --signal=TERM --kill-after=20s 180m taskset -c 0-3 nice -n 10 \
  "$TEACHER_PYTHON" -u scripts/serve_teacher.py --platform cpu --port "$PORT" --openpi-root "$OPENPI_ROOT" --checkpoint "$CHECKPOINT" \
  >"runtime/logs/${LABEL}-server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
for attempt in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "runtime/logs/${LABEL}-server.log" >&2
    exit 1
  fi
  if curl --max-time 2 -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
timeout --signal=TERM --kill-after=200s 170m taskset -c 4-7 nice -n 10 \
  bash scripts/sim_cpu.sh -u -m depth_policy.collect --config "$CONFIG" --split "$SPLIT" --attempts "$ATTEMPTS" "${EXTRA_ARGS[@]}" \
  >"runtime/logs/${LABEL}-collect.log" 2>&1 &
COLLECT_PID=$!
interrupt() {
  kill -TERM "$COLLECT_PID" 2>/dev/null || true
}
trap interrupt INT TERM
set +e
wait "$COLLECT_PID"
COLLECT_EXIT=$?
if kill -0 "$COLLECT_PID" 2>/dev/null; then
  wait "$COLLECT_PID"
  COLLECT_EXIT=$?
fi
set -e
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m depth_policy.summarize \
  --data "$DATA_DIR" --output "reports/${LABEL}-summary.json"
echo "Collector exit: $COLLECT_EXIT"
exit "$COLLECT_EXIT"
