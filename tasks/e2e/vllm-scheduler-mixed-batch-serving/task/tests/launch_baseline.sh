#!/usr/bin/env bash
# STRONG BASELINE — verifier-owned, well-tuned vLLM config for the target model on H20.
#
# This is the reward DENOMINATOR (the paired baseline). Under the bounded reward.md envelope a
# candidate that merely ties this configuration scores 0.
#
# 🔴 2026-07-28 — ONE knob given back, deliberately and once (owner ruling).
# The previous revision forced `--max-num-seqs 128`. With every configuration win priced in, the
# measured oracle ceiling was median-of-6 = 1.0269 against reward.md's >=1.15 authoring floor,
# and at that ref_speedup the log envelope is noise-dominated (MEASURED: a 1.0396 oracle run
# scored 0.732 against a 1.0269 constant — a +1.2% fluctuation moving reward by +0.23). So the
# task had no usable gradient.
#
# The knob given back is the ADMISSION WIDTH, and it is the least defensible as a default:
#   * vLLM's own `SchedulerConfig.max_num_seqs` default is None (i.e. the engine picks), so 128
#     is not "what you get" — it is an explicit, model- and GPU-specific tuning decision.
#     MEASURED via EngineArgs/SchedulerConfig introspection in-image, not assumed.
#   * every OTHER knob this baseline sets is either already the vLLM default (`enforce_eager`
#     False, `async_scheduling` False) or a correctness/capacity requirement rather than a
#     scheduling optimisation (`--dtype bfloat16`, `--max-model-len 8192`,
#     `--gpu-memory-utilization 0.88` vs the 0.9 default is a *reduction*, i.e. conservative).
#   * chunked prefill and prefix caching STAY ON, so the baseline remains a competent server a
#     reasonable engineer would ship — it is not degraded into the naive starter.
# Chosen value 64 rather than 32: at 32 the SHIPPED STARTER measures median-of-6 = 1.0101 > 1,
# i.e. discrimination breaks (the starter would earn reward). At 64 the starter measures 0.9316
# (bursts 0.8094) so it still scores 0, while the oracle measures 1.0504. Discrimination is
# preserved BY MEASUREMENT, not by assertion.
#
# The rest of the hardening is unchanged and still priced in: chunked prefill with a tuned chunk
# budget, CUDA graphs (NOT --enforce-eager), prefix caching, tuned GPU-memory utilization. To beat
# this the candidate must improve the SCHEDULER / continuous-batching itself — the prefill/decode
# interleave and chunk policy, the batching order, preemption/requeue, or the per-step token-budget
# accounting — not merely re-widen admission (a candidate that only raises --max-num-seqs recovers
# part of the gap and is welcome to; the hidden mix punishes a single knob, see the sweep record in
# tests/reward_manifest.json).
#
# FROZEN: uploaded fresh at scoring, sha256-pinned, never model-visible.

set -euo pipefail

PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/app/model}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Deterministic greedy path for the token-parity gate.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# 🔴 Every value below is a LITERAL. There is deliberately no env-var indirection and no
# variant switch: the verifier launches this script with the candidate's own os.environ
# inherited, so any `${SOMETHING:-default}` controlling a scheduling knob would let the
# candidate reconfigure its own baseline (i.e. lower the reward denominator) just by
# exporting a variable. PORT and MODEL_PATH are set by the harness itself and are the only
# permitted indirection.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name default \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests \
    --trust-remote-code
