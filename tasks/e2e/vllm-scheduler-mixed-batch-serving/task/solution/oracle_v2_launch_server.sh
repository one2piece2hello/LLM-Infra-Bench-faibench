#!/usr/bin/env bash
# REVIEWER-ONLY ORACLE (the strongest configuration found in the 2026-07-27 sweep of 13
# variants): the strong baseline's own flags PLUS async scheduling and 4 front-end API server
# processes. Measured median-of-6 paired ratio 1.0269 (bursts 1.0592); the provenance is recorded
# reviewer-side in the frozen manifest, deliberately NOT named here — the verifier's source-scan
# gate rejects any /app/submission file that references the evaluation surface, and it DID reject
# an earlier revision of this comment (a useful demonstration that the gate bites).
# Never baked into the image, never run at scoring time.
set -euo pipefail
PORT="${PORT:-30001}"
MODEL_PATH="${MODEL_PATH:-/app/model}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name default --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.88 \
    --max-model-len 8192 --max-num-seqs 128 --max-num-batched-tokens 8192 \
    --enable-chunked-prefill --enable-prefix-caching --disable-log-requests \
    --trust-remote-code --async-scheduling --api-server-count 4
