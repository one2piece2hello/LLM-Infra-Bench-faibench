#!/usr/bin/env bash
# /opt/loop/score_engine.sh — baked in-session DEV scoring engine for e2e-d1-varlen-prefill-attn-sol.
# Runs the candidate attention kernel on the PUBLIC dev suite, enforces the REAL bit-exact KV
# round-trip correctness gate, and reports the geomean achieved TFLOP/s as the best-of-k
# ranking signal. Called ONLY by /opt/loop/submit.sh. 0700 root-owned; the solver's uid cannot read
# this file or /opt/loop/private/**.
#
# 🔴 It bakes NO strong-baseline, NO hidden suite and NO calibrated anchor: the dev proxy uses the
# author's PUBLIC dev suite (a byte copy of the model-visible /app/dev_bench/dev_suite.json) and
# reports ABSOLUTE TFLOP/s, never normalized against the grade. Constants come from
# /opt/loop/private/manifest.json ONLY.
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
