#!/usr/bin/env bash
# Verifier entry — vLLM request-scheduling / continuous-batching serving optimization
# (E2E family B, H20).
#
# 2026-07-27: reward is now reward.md's BOUNDED [0,1] log envelope over the ABBA-paired
# MEDIAN speedup (it used to be an unbounded geomean in which tying the baseline was full marks).
# Every hard-fail short-circuit here passes a NAMED reward.md pre-gate to compute_reward.py,
# which emits the FULL 5-file /logs/verifier contract on every path.
#
# The frozen evaluation surface (this test.sh, compute_reward.py, launch_baseline.sh,
# prompts.jsonl, the timer daemon) is uploaded FRESH at scoring and is root-0700 /
# not model-visible, so the candidate cannot have tampered with it. The editable area
# is the WHOLE system (vLLM at /app/repo + site-packages + /app/submission) MINUS this
# frozen surface. Gate = (1) source-scan blocks references to verifier internals,
# (2) timer daemon must be alive (G2), (3) pristine vLLM restored before the baseline
# so the 1.0 anchor is measured against un-modified vLLM (the candidate re-applies its
# own patches from /app/submission at launch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-/app}"
VERIFIER_DIR="/logs/verifier"
mkdir -p "$VERIFIER_DIR"
# Login-style PATH so the venv python (vLLM) is found in the non-login scoring shell.
export PATH="/opt/kernelbench-venv/bin:${PATH}"

# --- 🔴 loop16 LEAK GATE ----------------------------------------------------------
# In the per-round development loop, /opt/loop/score_engine.sh tees this script's WHOLE
# stdout into a solver-readable /logs/loop/dev/verdict.raw and then reads
# /logs/verifier/reward.json. Un-redacted, those channels hand the solver the hidden
# workload names with their per-workload median_ms (baseline AND candidate), the gate
# thresholds with the measured rates, and the calibrated 1.0 anchor — every round. So when
# LOOP_DEV_OUT is set (only the loop sets it):
#   * the verifier's stdout+stderr go to a root-0600 private log, and
#   * an EXIT trap — which runs LAST, on every exit path including fail() — reduces
#     /logs/verifier to reward.txt + a reward.json carrying ONLY the score and a leak-free
#     failure CATEGORY the loop engine can still map to a named invariant.
# The authoritative end-of-session/single-shot path keeps the full diagnostic output.
LOOP_MODE=""
if [ -n "${LOOP_DEV_OUT:-}" ]; then LOOP_MODE=1; fi
VLOG="${VERIFIER_DIR}/.private_run.log"
if [ -n "$LOOP_MODE" ]; then : > "$VLOG"; chmod 600 "$VLOG" 2>/dev/null || true; fi
_run_verifier() {
    if [ -n "$LOOP_MODE" ]; then "$@" >> "$VLOG" 2>&1; else "$@"; fi
}
_loop_redact() {
    if [ -z "$LOOP_MODE" ]; then return 0; fi
    python3 - "$VERIFIER_DIR" <<'REDACTEOF'
import glob, json, os, shutil, sys
VD = sys.argv[1]
try:
    p = json.load(open(os.path.join(VD, "reward.json")))
except Exception:
    p = {}
score = float(p.get("score") or p.get("reward") or 0.0)
reason = str((p.get("additional_data") or {}).get("reason") or "").lower()


def category(r):
    """Leak-free category. The strings are chosen so /opt/loop/score_engine.sh still maps
    them to the right named invariant (out_of_scope_edit / timing_invalid /
    correctness_trace_mismatch / baseline_timing_failed / candidate_timing_failed)."""
    if not r:
        return "verifier_completed"
    if ("frozen" in r or "verifier internals" in r or "source references" in r
            or "shadow" in r or "digest" in r):
        return "frozen surface / source references the evaluation surface"
    if "timer" in r:
        return "timer daemon integrity"
    if ("token match" in r or "parity" in r or "conform" in r or "correctness" in r
            or "attribution" in r or "contaminat" in r or "degenerate" in r
            or "identity" in r or "truncat" in r or "work token" in r):
        return "token match / parity gate not met"
    if "baseline" in r and "valid outputs" in r:
        return "baseline produced no valid outputs"
    if ("server" in r or "verifier error" in r or "launch" in r or "timeout" in r
            or "not found" in r or "missing" in r):
        return "candidate server error"
    if "complete" in r:
        return "benchmark complete"
    return "verifier_completed"


red = {"score": score, "reward": score, "subscores": [],
       "additional_data": {
           "reason": category(reason),
           "note": "REDACTED for the per-round development loop: hidden workload identities, "
                   "per-workload timings, gate thresholds and the calibrated baseline are part "
                   "of the evaluation surface and are not disclosed. The end-of-session score "
                   "is authoritative."}}
json.dump(red, open(os.path.join(VD, "reward.json"), "w"), indent=1)
with open(os.path.join(VD, "reward.txt"), "w") as fh:
    fh.write("%s\n" % score)
for f in glob.glob(os.path.join(VD, "*")) + glob.glob(os.path.join(VD, ".*")):
    if os.path.basename(f) in ("reward.json", "reward.txt", ".", ".."):
        continue
    try:
        if os.path.isdir(f):
            shutil.rmtree(f, ignore_errors=True)
        else:
            os.remove(f)
    except Exception:
        pass
REDACTEOF
}
trap _loop_redact EXIT
PY_RUN=(python3)

HARBOR_START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
# fail <human message> [named reward.md pre-gate]
fail() {
    _run_verifier "${PY_RUN[@]}" "${SCRIPT_DIR}/compute_reward.py" --fail "$1" \
        --fail-gate "${2:-build_or_entry_contract_failed}" \
        --total-time-ms "$(( $(python3 -c "import time; print(int(time.time()*1000))") - HARBOR_START_MS ))" \
        --output-dir "$VERIFIER_DIR"
    exit 0
}

# /logs/verifier carries the hidden workload names, their timings, the frozen ref_speedup and
# correctness mismatch prefixes. The harness reads it as root; nothing else needs to.
chmod 0700 "$VERIFIER_DIR" 2>/dev/null || true

echo "=== vLLM Scheduling / Continuous-Batching Serving Optimization — Verifier ==="

# The FROZEN reward manifest (ref_speedup + every threshold) is uploaded FRESH with tests/.
[ -f "${SCRIPT_DIR}/reward_manifest.json" ] || \
    fail "frozen reward manifest missing at ${SCRIPT_DIR}/reward_manifest.json" build_or_entry_contract_failed
echo "PASS: frozen reward manifest present"

# --- Pre-flight ---
[ -f "${APP_DIR}/submission/launch_server.sh" ] || fail "submission/launch_server.sh not found" build_or_entry_contract_failed
echo "PASS: submission/launch_server.sh exists"
{ [ -d "${APP_DIR}/model" ] && [ -f "${APP_DIR}/model/config.json" ]; } || fail "Model weights not found at /app/model" build_or_entry_contract_failed
echo "PASS: model weights present"

# --- Timer-integrity gate (G2) ---------------------------------------------------------
# 🔴 2026-07-27: the previous shape only checked that the pid in /app/.timer/timer.pid was
# alive. That gate is UNTRIPPABLE and therefore decorative: BASH_ENV=/etc/profile.d/
# frontier-task-init.sh restarts the daemon on the next non-interactive bash — i.e. on
# `bash /tests/test.sh` itself — and rewrites timer.pid. MEASURED: two tampers (kill the pid;
# `chmod 000 /app/timer.sh` then kill) both left the gate green through a full 840 s run.
# A permanently-green gate reads as protection it does not provide, so it is replaced by
# checks of the properties that actually matter AND that a candidate can actually violate:
#   (a) the published budget must equal the harness's own TASK_BUDGET_SECS  — a candidate that
#       rewrites /app/.timer/budget_secs to buy itself more time is caught;
#   (b) /app/timer.sh must match the sha256 the frozen manifest pins — a candidate that
#       rewrites the daemon (to publish a frozen or slowed clock) is caught;
#   (c) the clock must still ADVANCE — a stopped/frozen clock is caught.
TIMER_DIR="${APP_DIR}/.timer"
if [ -f "${SCRIPT_DIR}/reward_manifest.json" ]; then
    WANT_TIMER_SHA=$(python3 -c "import json;print((json.load(open('${SCRIPT_DIR}/reward_manifest.json')).get('frozen_sha256') or {}).get('/app/timer.sh',''))" 2>/dev/null)
    if [ -n "$WANT_TIMER_SHA" ] && [ -f "${APP_DIR}/timer.sh" ]; then
        GOT_TIMER_SHA=$(sha256sum "${APP_DIR}/timer.sh" | awk '{print $1}')
        [ "$GOT_TIMER_SHA" = "$WANT_TIMER_SHA" ] || \
            fail "timer daemon binary modified (sha256 mismatch)" cheating_detected
        echo "PASS: timer daemon sha256 matches the frozen manifest"
    fi
fi
if [ -n "${TASK_BUDGET_SECS:-}" ] && [ -f "${TIMER_DIR}/budget_secs" ]; then
    PUB_BUDGET=$(tr -dc '0-9' < "${TIMER_DIR}/budget_secs")
    if [ -n "$PUB_BUDGET" ] && [ "$PUB_BUDGET" != "$TASK_BUDGET_SECS" ]; then
        fail "published wall-clock budget ${PUB_BUDGET}s != the harness budget ${TASK_BUDGET_SECS}s (timer tampered)" cheating_detected
    fi
    echo "PASS: published budget matches the harness budget"
fi
# (c) the published elapsed time must AGREE with the real elapsed time derived from the
#     daemon's own start_epoch. This is exact and needs no sleep, so it cannot flake: MEASURED,
#     the daemon ticks every 10 s, so a naive "did elapsed_secs change over 3 s?" probe
#     false-positives on a perfectly healthy clock. A frozen or slowed clock diverges from
#     `now - start_epoch` without bound and is caught on the first check.
if [ -f "${TIMER_DIR}/elapsed_secs" ] && [ -f "${TIMER_DIR}/start_epoch" ]; then
    T_PUB=$(tr -dc '0-9' < "${TIMER_DIR}/elapsed_secs")
    T_START=$(tr -dc '0-9' < "${TIMER_DIR}/start_epoch")
    if [ -n "$T_PUB" ] && [ -n "$T_START" ]; then
        T_REAL=$(( $(date +%s) - T_START ))
        T_SKEW=$(( T_REAL - T_PUB )); [ "$T_SKEW" -lt 0 ] && T_SKEW=$(( -T_SKEW ))
        # tolerance covers the 10 s tick plus scheduling jitter
        if [ "$T_SKEW" -gt 60 ]; then
            fail "published elapsed ${T_PUB}s disagrees with the real elapsed ${T_REAL}s by ${T_SKEW}s (wall clock frozen, slowed or rewritten)" cheating_detected
        fi
        echo "PASS: published clock agrees with real elapsed time (skew ${T_SKEW}s)"
    fi
fi

# --- Source-scan gate: block references to the frozen verifier internals ---
SUSPICIOUS='/tests/|compute_reward|launch_baseline|/logs/verifier|reward\.json|reward\.txt|prompts\.jsonl|reward_manifest|/opt/loop/private|hidden_suite'
while IFS= read -r -d '' f; do
    if grep -q -E "${SUSPICIOUS}" "$f" 2>/dev/null; then
        fail "Source references verifier internals: $f" forbidden_edit_path
    fi
done < <(find "${APP_DIR}/submission" -type f \
    \( -name "*.py" -o -name "*.sh" -o -name "*.json" -o -name "*.txt" \
       -o -name "*.toml" -o -name "*.yaml" -o -name "*.yml" -o -name "*.cfg" \) \
    -not -path "*/\.*" -print0 2>/dev/null)
echo "PASS: source scan (submission)"

# Oracle marker (validation only).
ORACLE_FLAG=""
[ -f "${APP_DIR}/.oracle_solution" ] && { ORACLE_FLAG="--oracle"; echo "INFO: oracle marker detected"; }

# --- Restore pristine vLLM so the BASELINE is measured against un-modified vLLM. ---
# The candidate may edit vLLM site-packages (editable area); this restore guarantees a
# clean 1.0 anchor. The candidate re-applies its own patches from /app/submission at launch.
SITE_PKG=$(python3 -c "import vllm,os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null || true)
if [ -n "$SITE_PKG" ] && [ -f "${APP_DIR}/.vllm-baseline.tar" ]; then
    tar xf "${APP_DIR}/.vllm-baseline.tar" -C "$SITE_PKG" 2>/dev/null || true
    find "$SITE_PKG/vllm" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$SITE_PKG/vllm" -name "*.pyc" -delete 2>/dev/null || true
    rm -rf /root/.triton/cache "${HOME:-/root}/.cache/vllm" 2>/dev/null || true
    echo "PASS: restored pristine vLLM + cleared caches"
else
    echo "WARN: no vLLM snapshot; baseline may use candidate-modified vLLM"
fi

# Kill leftover GPU processes from the agent so the verifier starts fresh servers.
pkill -f "vllm" 2>/dev/null || true
pkill -f "python.*api_server" 2>/dev/null || true
sleep 2
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

HARBOR_TOTAL_MS=$(( $(python3 -c "import time; print(int(time.time()*1000))") - HARBOR_START_MS ))
_run_verifier "${PY_RUN[@]}" "${SCRIPT_DIR}/compute_reward.py" \
    --app-dir "${APP_DIR}" \
    --output-dir "$VERIFIER_DIR" \
    --total-time-ms "$HARBOR_TOTAL_MS" \
    ${ORACLE_FLAG}

# Contract floor: if compute_reward.py died before writing anything, the 5-file contract
# must still exist (a sibling task shipped a scorer that wrote 0 files on a hard crash).
if [ ! -f "$VERIFIER_DIR/reward.txt" ]; then
    for f in reward.json metrics.json verifier_state.json correctness_results.json benchmark_results.json; do
        printf '{"task_type":"performance","reward":0.0,"hard_fail_reasons":["build_or_entry_contract_failed"],"speedup":null,"ref_speedup":null,"passed":0,"total":0}\n' \
            > "$VERIFIER_DIR/$f"
    done
    echo 0.0 > "$VERIFIER_DIR/reward.txt"
    echo "WARN: emitted the contract floor (scorer produced no output)"
fi

echo ""
echo "=== Verifier complete ==="
[ -f "$VERIFIER_DIR/reward.txt" ] && echo "Score: $(cat "$VERIFIER_DIR/reward.txt")"
