#!/usr/bin/env bash
# /opt/loop/score_engine.sh — baked in-session DEV scoring engine for
# e2e-vllm-scheduler-mixed-batch-serving. Launches the candidate's vLLM server, confirms it serves
# the PUBLIC dev prompts (a liveness/usability correctness proxy), and reports serving throughput
# (1000/median_ms) as the best-of-k ranking signal. Called ONLY by /opt/loop/submit.sh. 0700
# root-owned; the solver's uid cannot read this file or /opt/loop/private/**.
#
# 🔴 It bakes NO strong baseline, NO hidden prompts and NO calibrated anchor: the dev proxy uses the
# same PUBLIC prompts as the model-visible /app/run_dev_bench.py and reports ABSOLUTE throughput,
# never normalized against the grade. Constants come from /opt/loop/private/manifest.json ONLY.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail
git config --global --add safe.directory '*' 2>/dev/null || true

DEV_OUT=/logs/loop/dev
mkdir -p "$DEV_OUT"
rm -f "$DEV_OUT/verifier_state.json" "$DEV_OUT/reward.json" "$DEV_OUT/harness_error.txt" 2>/dev/null || true

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "harness_error: no python on PATH" > "$DEV_OUT/harness_error.txt"; exit 3
fi

# clear leftover GPU servers from a previous round so this round boots fresh
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "api_server" 2>/dev/null || true
sleep 2

"$PY" /opt/loop/private/tests/dev_eval.py
rc=$?

if [ ! -s "$DEV_OUT/verifier_state.json" ] && [ ! -s "$DEV_OUT/harness_error.txt" ]; then
  echo "harness_error: dev_eval.py died before writing a result (rc=$rc)" > "$DEV_OUT/harness_error.txt"
  exit 3
fi
if [ -s "$DEV_OUT/harness_error.txt" ]; then
  exit 3
fi
exit 0
