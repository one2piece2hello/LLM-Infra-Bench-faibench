#!/usr/bin/env bash
# REVIEWER-ONLY candidate RELAXED strong baselines for e2e-vllm-scheduler-mixed-batch-serving.
# Never baked into the image, never part of tests/ in this form — the CHOSEN variant is
# materialised into tests/launch_baseline.sh at calibration time.
#
# WHY: the shipped strong baseline priced in EVERY configuration win, which left the measured
# oracle ceiling at median-of-6 = 1.0269 — far below reward.md's 1.15 authoring floor, and at a
# ref_speedup that close to 1 the log envelope is noise-dominated (a +1.2% fluctuation moved a
# measured reward from 0.5 to 0.732). The remedy chosen by the owner is to give ONE tuned knob
# back, so the task has real headroom again.
#
# Each variant keeps the baseline HONEST (a correct, non-degenerate, well-configured server that
# a competent engineer would plausibly ship) and gives back exactly one knob:
#   seqs32  : --max-num-seqs 32   (vLLM's own default admission width; the baseline forced 128)
#   seqs64  : --max-num-seqs 64   (a milder version of the same)
#   tok2048 : --max-num-batched-tokens 2048 (a small chunked-prefill token budget; the baseline
#             forced 8192). Chunked prefill stays ON, so the baseline is still not naive.
set -euo pipefail

PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/app/model}"
R="${RELAX:-seqs32}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

ARGS=(
    --model "$MODEL_PATH"
    --served-model-name default
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --gpu-memory-utilization 0.88
    --max-model-len 8192
    --enable-chunked-prefill
    --enable-prefix-caching
    --disable-log-requests
    --trust-remote-code
)

case "$R" in
  seqs32)  ARGS+=(--max-num-seqs 32  --max-num-batched-tokens 8192) ;;
  seqs64)  ARGS+=(--max-num-seqs 64  --max-num-batched-tokens 8192) ;;
  tok2048) ARGS+=(--max-num-seqs 128 --max-num-batched-tokens 2048) ;;
  asis)    ARGS+=(--max-num-seqs 128 --max-num-batched-tokens 8192) ;;
  *) echo "unknown RELAX=$R" >&2; exit 2 ;;
esac

echo "[relaxed-baseline] variant=$R" >&2
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
