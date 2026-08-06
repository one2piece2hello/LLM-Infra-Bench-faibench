#!/usr/bin/env bash
# DEV scoring engine (0700 root:root) for compile-cache-key-canonicalize. Invoked ONLY by
# /opt/loop/submit.sh — never by the solver.
#
# It reuses the task verifier VERBATIM: it runs the baked copy of the base
# task's tests/test.sh (byte-identical except its output dir is /logs/loop/dev
# instead of /logs/verifier) against the LIVE /app/repo in the default candidate
# mode (which scores /app/repo AS-IS and never clobbers it). The base
# compute_reward.py writes reward = speedup (uncapped). The authoritative reward
# still comes from the runner's end-of-session /tests/ run, NOT from here.
#
# PATH-pin: the agent shell + runner verifier exec are
# NON-login; bare python3 has no torch. Pin the venv FIRST.
export PATH=/opt/kernelbench-venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_OUT="${LOOP_DEV_OUT:-/logs/loop/dev}"
mkdir -p "$DEV_OUT"
# fresh run: clear any stale harness marker
rm -f "$DEV_OUT/harness_error.txt" 2>/dev/null || true

# Run the baked task verifier (candidate mode). Suppress its stdout entirely
# (the base test.sh echoes a summary line carrying ref_speedup / hardware / raw
# speedup — that MUST NOT reach the solver; only sanitize_feedback.py emits).
KERNELBENCH_VERIFY_MODE=candidate bash "$HERE/private/dev_tests/test.sh" \
  > "$DEV_OUT/dev_engine_stdout.log" 2>&1
rc=$?

# If the base verifier could not even write a reward (infra/torch/cuda failure),
# it exits non-zero WITHOUT a reward.json. Synthesize a harness marker so
# submit.sh treats it as infra (does not consume the solver's budget) rather than
# a silent candidate FAIL.
if [ "$rc" -ne 0 ] && [ ! -f "$DEV_OUT/reward.json" ] && [ ! -f "$DEV_OUT/harness_error.txt" ]; then
  {
    echo "dev_engine_exit_${rc}"
    tail -n 40 "$DEV_OUT/dev_engine_stdout.log" 2>/dev/null
  } > "$DEV_OUT/harness_error.txt"
  exit 3
fi
exit 0
