#!/usr/bin/env bash
# /opt/loop/score_engine.sh — the baked in-session DEV scoring engine for
# e2e-h3-eval-harness-throughput-quality. Runs the candidate scoring pipeline on the PUBLIC dev
# split, enforces the REAL welded consistency + anti-cache gates, and reports a RAW speedup vs the
# PUBLIC naive template as the best-of-k ranking signal. Called ONLY by /opt/loop/submit.sh.
# 0700 root-owned; the solver's uid cannot read this file or /opt/loop/private/**.
#
# 🔴 It bakes NO strong-baseline reference, NO held-out sample set and NO calibrated anchor: the dev
# proxy divides by the PUBLIC naive template (which already ships in the image), never by the hidden
# strong baseline, and never normalizes against ref_speedup. Constants come from
# /opt/loop/private/manifest.json ONLY (never from the solver-controlled environment).
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail

DEV_OUT=/logs/loop/dev
mkdir -p "$DEV_OUT"
# clear the previous round's artifacts so a stale file never masquerades as this round
rm -f "$DEV_OUT/verifier_state.json" "$DEV_OUT/reward.json" "$DEV_OUT/harness_error.txt" 2>/dev/null || true

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "harness_error: no python on PATH" > "$DEV_OUT/harness_error.txt"
  exit 3
fi

# Pin thread counts so the (cheap) dev pass does not depend on node core-count; the algorithmic
# throughput gradient (candidate vs naive template) persists regardless.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES=""

"$PY" /opt/loop/private/tests/dev_eval.py
rc=$?

# dev_eval writes its own JSON on every path it reaches. If the interpreter itself died (missing dep
# / OOM / SIGKILL) nothing was written -> classify as harness_error so the round is refunded rather
# than mis-scored as a candidate defect.
if [ ! -s "$DEV_OUT/verifier_state.json" ] && [ ! -s "$DEV_OUT/harness_error.txt" ]; then
  echo "harness_error: dev_eval.py died before writing a result (rc=$rc)" > "$DEV_OUT/harness_error.txt"
  exit 3
fi
# propagate the refund signal if dev_eval flagged an infra failure
if [ -s "$DEV_OUT/harness_error.txt" ]; then
  exit 3
fi
exit 0
