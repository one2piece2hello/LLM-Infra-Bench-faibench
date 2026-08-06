#!/usr/bin/env bash
# /opt/loop/score_engine.sh — the baked in-session DEV scoring engine for
# e2e-a3-moe-train-budget. Runs the candidate recipe on a SMALL public token
# shard under the uid-65534 drop and scores an ABSOLUTE public-proxy val_bpb.
# Called ONLY by /opt/loop/submit.sh. 0700 root-owned; the solver's dropped uid cannot
# read this file or /opt/loop/private/**.
#
# 🔴 It reads its constants from /opt/loop/private/manifest.json ONLY — never from the
# environment. The loop runs in the solver-writable session shell, so anything read from
# env is attacker-controlled; in particular the reward's anchor is NOT in this container
# at all (it is uploaded fresh to /tests at grading), and this engine never reads /tests.
set -uo pipefail
# The agent invokes submit.sh from a NON-login shell, so bare python3 has no torch.
# Pin the kernelbench venv onto PATH so python AND the training subprocess have torch.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

DEV_OUT=/logs/loop/dev
mkdir -p "$DEV_OUT"
# clear the previous round's artifacts so a stale file never masquerades as this round
rm -f "$DEV_OUT/verifier_state.json" "$DEV_OUT/reward.json" "$DEV_OUT/harness_error.txt" 2>/dev/null || true

PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "harness_error: no python on PATH" > "$DEV_OUT/harness_error.txt"
  exit 3
fi

# Pin thread counts so the (cheap) eval does not depend on node core-count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

"$PY" /opt/loop/private/dev_eval.py
rc=$?

# dev_eval writes its own JSON on every path it reaches. If the interpreter itself died
# (missing dep / OOM / SIGKILL) nothing was written -> classify as harness_error so the
# round is refunded rather than mis-scored as a candidate defect.
if [ ! -s "$DEV_OUT/verifier_state.json" ] && [ ! -s "$DEV_OUT/harness_error.txt" ]; then
  echo "harness_error: dev_eval.py died before writing a result (rc=$rc)" > "$DEV_OUT/harness_error.txt"
  exit 3
fi
# propagate the refund signal if dev_eval flagged an infra failure
if [ -s "$DEV_OUT/harness_error.txt" ]; then
  exit 3
fi
exit 0
