#!/usr/bin/env bash
# CANDIDATE ENTRYPOINT — you own and modify this file (and anything under /app/submission).
#
# The verifier runs this to start your server. It receives:
#   PORT       — the port your OpenAI-compatible server must listen on
#   MODEL_PATH — the path to the model weights (/app/model)
#
# STARTING POINT: a plain vLLM server with default scheduling and a small batch width and
# no explicit chunked-prefill / prefix-caching tuning — an honest, correct server that will
# score BELOW the strong baseline (which uses a tuned chunked-prefill + admission config).
# Your job is to make it FASTER on the hidden mixed-burst workloads while keeping greedy
# outputs identical.
#
# To beat the strong baseline you must go beyond configuration — improve the request
# scheduler / continuous-batching itself (admission control, prefill/decode interleave and
# chunking policy, batching order, preemption/requeue, token-budget accounting), write
# custom kernels, or modify vLLM source. Anything you need at verification time (helper
# modules, source patches, kernels, config) must live under /app/submission and be
# (re)applied here before the server starts. Direct edits to site-packages alone may be
# lost during replay.
set -euo pipefail

PORT="${PORT:-30001}"
MODEL_PATH="${MODEL_PATH:-/app/model}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# --- (re)apply your optimizations here, e.g. patch vLLM source from /app/submission ---
# cp /app/submission/my_scheduler.py "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/v1/core/sched/scheduler.py"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name default \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --disable-log-requests \
    --trust-remote-code
