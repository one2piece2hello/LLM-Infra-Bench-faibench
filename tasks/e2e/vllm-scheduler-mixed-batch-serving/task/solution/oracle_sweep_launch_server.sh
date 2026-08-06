#!/usr/bin/env bash
# REVIEWER-ONLY oracle candidate launcher (sweep harness) for
# e2e-vllm-scheduler-mixed-batch-serving. NEVER baked into the image, NEVER part
# of tests/, NEVER run at scoring time.
#
# Selects a candidate ORACLE configuration via ORACLE_VARIANT. The strong baseline
# (tests/launch_baseline.sh) already prices in chunked prefill + a tuned admission
# width + prefix caching + CUDA graphs + tuned GPU-memory utilisation, so every
# variant here starts from EXACTLY those flags and adds one scheduler-subsystem
# lever on top; the point is to find whether real headroom over the hardened
# baseline exists at all, and how big it is.
set -euo pipefail

PORT="${PORT:-30001}"
MODEL_PATH="${MODEL_PATH:-/app/model}"
V="${ORACLE_VARIANT:-v1_async}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# The baseline's flag set, verbatim (the 1.0 anchor's configuration).
BASE_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name default
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --gpu-memory-utilization 0.88
    --max-model-len 8192
    --max-num-seqs 128
    --max-num-batched-tokens 8192
    --enable-chunked-prefill
    --enable-prefix-caching
    --disable-log-requests
    --trust-remote-code
)
EXTRA=()

case "$V" in
  v0_selfcheck)   ;;                                     # baseline-as-candidate: expect ~1.0
  v1_async)       EXTRA=(--async-scheduling) ;;
  v2_async_api4)  EXTRA=(--async-scheduling --api-server-count 4) ;;
  v3_async_fullcg)
      EXTRA=(--async-scheduling --compilation-config '{"full_cuda_graph": true}') ;;
  v4_async_bigtok)
      EXTRA=(--async-scheduling --max-num-batched-tokens 16384 --disable-log-stats) ;;
  v5_api4_only)   EXTRA=(--api-server-count 4 --disable-log-stats) ;;
  v6_combo)
      EXTRA=(--async-scheduling --api-server-count 4 --disable-log-stats
             --max-num-batched-tokens 16384
             --cuda-graph-sizes 1 2 4 8 16 24 32 48 64 96 128 160 192 256) ;;
  v7_async_cg128)
      EXTRA=(--async-scheduling --disable-log-stats
             --cuda-graph-sizes 1 2 4 8 16 24 32 48 64 96 128 160 192 256) ;;
  # --- round 2: SPECULATIVE DECODING -------------------------------------------------
  # The bursts run at ~8% of the H20 weight-bandwidth roofline (13.5k tok/s for a 1.5B
  # bf16 model), i.e. they are PER-STEP-OVERHEAD bound, not compute bound. n-gram / prompt-
  # lookup speculation cuts the number of engine steps, which is exactly the bound
  # resource. Under greedy decoding (temperature 0) vLLM's rejection sampling is
  # output-LOSSLESS, so the token-parity gate is unaffected.
  v8_ngram)
      EXTRA=(--disable-log-stats --speculative-config
             '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}') ;;
  v9_api4_ngram)
      EXTRA=(--api-server-count 4 --disable-log-stats --speculative-config
             '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}') ;;
  v10_async_api4_ngram)
      EXTRA=(--async-scheduling --api-server-count 4 --disable-log-stats --speculative-config
             '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}') ;;
  v11_api4_seqs256)
      EXTRA=(--api-server-count 4 --disable-log-stats --max-num-seqs 256) ;;
  v12_api4_ngram3)
      EXTRA=(--api-server-count 4 --disable-log-stats --speculative-config
             '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}') ;;
  v13_api8_ngram)
      EXTRA=(--api-server-count 8 --disable-log-stats --speculative-config
             '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}') ;;
  *) echo "unknown ORACLE_VARIANT=$V" >&2; exit 2 ;;
esac

# --max-num-batched-tokens appears twice when a variant overrides it; vLLM's argparse
# keeps the LAST occurrence, which is the variant's value. Order matters, so the
# variant flags come last.
echo "[oracle] variant=$V flags: ${EXTRA[*]:-<baseline only>}" >&2
exec python3 -m vllm.entrypoints.openai.api_server "${BASE_ARGS[@]}" "${EXTRA[@]}"
