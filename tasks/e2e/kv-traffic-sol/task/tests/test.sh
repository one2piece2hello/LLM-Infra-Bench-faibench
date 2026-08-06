#!/bin/bash
# e2e-b1-kv-traffic-sol — verifier entry (Type-3 E2E, open-ended metric).
# 7-step canonical structure with the E2E frozen-surface gate.
# The frozen surface (this script + compute_reward.py + harness/ + hidden_suite.json + the
# strong baseline + the manifest) is uploaded FRESH under /tests at scoring; a root-0700 copy
# at /opt/verifier is the fallback. reward 0 on ANY hard fail.
set -o pipefail

# --- step 0: PATH pin (the harness may exec login-style while the agent shell is NON-login — a bare
#     python3 must still resolve to the venv that has torch/triton/vllm). ---
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY="$(command -v python3 || command -v python)"

SUBMISSION_DIR="${SUBMISSION_DIR:-/app/repo/submission}"
REPO_DIR="${REPO_DIR:-/app/repo}"
MODE="${VERIFIER_MODE:-candidate}"     # candidate | strong_baseline | negative_* | ceiling_*
mkdir -p /logs/verifier /logs/artifacts "${SUBMISSION_DIR}"
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/verifier_state.json

emit_zero() {
    _d="$1"
    _r="${2:-build_or_entry_contract_failed}"
    # 🔴 LOOP mode: the reason string names frozen-surface files / hidden case ids and
    #    this early-exit path never reaches the redaction step below, so reduce it here.
    [ -n "${LOOP_DEV_OUT:-}" ] && _d="verifier_gate_failed"
    echo "HARD-FAIL: ${_d}"
    printf '{"task_type":"performance","reward":0.0,"hard_fail_reasons":["%s"],"speedup":null,"ref_speedup":null,"cv":{},"quality_gate_passed":false,"detail":"%s"}\n' "${_r}" "${_d}" > /logs/verifier/reward.json
    printf '{"task_type":"performance","reward":0.0,"hard_fail_reasons":["%s"],"quality_gate_passed":false,"detail":"%s"}\n' "${_r}" "${_d}" > /logs/verifier/verifier_state.json
    printf '{"metric":"log_speedup_vs_ref_speedup","reward":0.0,"speedup":0.0,"ref_speedup":null,"candidate_geomean_sol_fraction":0.0,"strong_baseline_geomean_sol_fraction":0.0,"metric_void_on_hard_fail":true,"detail":"%s"}\n' "${_d}" > /logs/verifier/benchmark_results.json
    printf '{"quality_gate":"not reached","passed":false,"detail":"%s"}\n' "${_d}" > /logs/verifier/correctness_results.json
    echo 0.000000 > /logs/verifier/reward.txt
}
save_artifacts() {
    # 🔴 This EXIT trap runs LAST — i.e. AFTER the LOOP-mode redaction below — so in LOOP
    #    mode it must publish nothing at all (/logs/artifacts is solver-readable).
    if [ -n "${LOOP_DEV_OUT:-}" ]; then
        rm -rf /logs/artifacts/* 2>/dev/null || true
        return 0
    fi
    for f in kv_traffic.py notes.md action.log; do
        [ -e "${SUBMISSION_DIR}/${f}" ] && cp -r "${SUBMISSION_DIR}/${f}" "/logs/artifacts/${f}" 2>/dev/null || true
    done
}
trap save_artifacts EXIT

# --- step 1: locate the FRESH frozen surface (never a baked copy when /tests is present). ---
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
TESTS_DIR="${SELF_DIR}"
if [ ! -f "${TESTS_DIR}/compute_reward.py" ]; then TESTS_DIR="/tests"; fi
if [ ! -f "${TESTS_DIR}/compute_reward.py" ]; then TESTS_DIR="/opt/verifier"; fi
VERIFIER="${TESTS_DIR}/compute_reward.py"
MANIFEST="${TESTS_DIR}/verifier-correctness-manifest.json"
if [ ! -f "${VERIFIER}" ]; then emit_zero "verifier missing"; exit 1; fi
for f in harness/bench_kvtraffic.py harness/baseline_kv_traffic.py harness/hidden_suite.json; do
    if [ ! -f "${TESTS_DIR}/${f}" ]; then emit_zero "frozen surface incomplete: ${f}"; exit 1; fi
done

# --- step 2: FROZEN-SURFACE HARD GATE.
#     The whole repo is editable EXCEPT the evaluation surface. Any touch => reward 0. ---
if [ -f "${MANIFEST}" ]; then
    bad=$("${PY}" - "$TESTS_DIR" "$MANIFEST" <<'EOF'
import hashlib, json, os, sys
tests, man_path = sys.argv[1], sys.argv[2]
man = json.load(open(man_path))
bad = []
for rel, exp in (man.get("frozen_surface_sha256") or {}).items():
    p = os.path.join(tests, rel)
    if not os.path.exists(p):
        bad.append("%s:MISSING" % rel); continue
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if exp and got != exp:
        bad.append("%s:SHA_MISMATCH" % rel)
print(";".join(bad))
EOF
)
    if [ -n "${bad}" ]; then emit_zero "frozen surface tampered: ${bad}" "forbidden_edit_path"; exit 1; fi
fi
# diagnostic diff against the baked single commit (only frozen paths are protected)
if [ -d "${REPO_DIR}/.git" ]; then
    ( cd "${REPO_DIR}" && git status --porcelain -uall > /logs/artifacts/repo_status.txt 2>/dev/null ) || true
    ( cd "${REPO_DIR}" && git diff --name-only HEAD > /logs/artifacts/repo_diff.txt 2>/dev/null ) || true
fi
for p in /tests /opt/verifier /opt/negative /app/timer.sh /app/.timer; do
    if [ -e "$p" ] && [ -w "$p" ] && [ "$(id -u)" != "0" ]; then
        emit_zero "frozen surface writable by the solver: $p" "forbidden_edit_path"; exit 1
    fi
done

# --- step 3: entry contract + import-origin assert (the entry must import from the baked repo). ---
if [ "${MODE}" = "candidate" ] && [ ! -f "${SUBMISSION_DIR}/kv_traffic.py" ]; then
    emit_zero "missing ${SUBMISSION_DIR}/kv_traffic.py (entry contract)"; exit 1
fi
"${PY}" - <<'EOF' || { echo "IMPORT-ORIGIN WARN"; }
import sys
try:
    import torch, triton
    print("[origin] torch", torch.__version__, "triton", triton.__version__)
    import vllm
    print("[origin] vllm", vllm.__version__, vllm.__file__)
    assert vllm.__file__.startswith("/app/repo"), "vllm must import from /app/repo"
except Exception as e:
    print("[origin] FAIL", e); sys.exit(1)
EOF

# --- step 4: timer daemon liveness (harness-owned clock, §1 G2). ---
if [ -f /app/timer.sh ] && [ ! -f /app/.timer/elapsed ]; then
    echo "[timer] WARN: /app/.timer/elapsed absent"
fi

# --- step 5/6/7: the correctness gate (bit-exact round-trip + poison + alias + plan + budget)
#     runs INSIDE the harness and is a hard prerequisite; only then is the traffic timed; then
#     the open-ended reward is computed. compute_reward.py emits the 5-file contract. ---
export VERIFIER_MODE="${MODE}"
export SUBMISSION_DIR
cd /tmp || exit 1
# 🔴 In LOOP mode the whole stdout of THIS script is teed by /opt/loop/score_engine.sh
#    into a solver-readable /logs/loop/dev/verdict.raw, and the verifier prints the
#    candidate AND baseline metric (the calibrated 1.0 anchor) plus per-case detail.
#    Send it to a private root-0600 log instead; only the one-line verdict JSON below
#    reaches stdout in LOOP mode.
if [ -n "${LOOP_DEV_OUT:-}" ]; then
    VLOG=/logs/verifier/.private_run.log
    : > "${VLOG}"; chmod 600 "${VLOG}" 2>/dev/null || true
    "${PY}" "${VERIFIER}" > "${VLOG}" 2>&1
else
    "${PY}" "${VERIFIER}"
fi
rc=$?
if [ ! -f /logs/verifier/reward.txt ]; then emit_zero "verifier produced no reward.txt (rc=${rc})"; exit 1; fi
if [ -z "${LOOP_DEV_OUT:-}" ]; then
    echo "[test.sh] done rc=${rc} reward=$(cat /logs/verifier/reward.txt)"
else
    echo "[test.sh] done rc=${rc}"
fi
# --- loop16 integration: emit the one-line verdict JSON the submission-loop scoring engine
#     parses from stdout (keys "gates" + "hard_fails"). Harmless for the single-shot path.
#     🔴 In LOOP mode the hard-fail reason is reduced to a leak-free CATEGORY and the verifier's
#     own 5-file output is REDACTED in place: a per-round /logs/verifier left intact would hand
#     the solver the hidden case ids, the parity tolerances, the per-case sol_fractions and the
#     measured hardware peak (the metric internals §7 forbids disclosing). The authoritative
#     single-shot scoring path (no LOOP_DEV_OUT) keeps the full diagnostic output. ---
"${PY}" - <<'PYEOF'
import glob, json, os
LOOP = bool(os.environ.get("LOOP_DEV_OUT"))
try:
    r = json.load(open('/logs/verifier/reward.json'))
except Exception:
    r = {"reward": 0.0, "quality_gate_passed": False, "detail": "no reward.json"}
ok = bool(r.get("quality_gate_passed"))
rew = float(r.get("reward") or 0.0)
sp = r.get("speedup")
sp = float(sp) if isinstance(sp, (int, float)) else 0.0
detail = str(r.get("detail") or "")
spec_fails = [str(x) for x in (r.get("hard_fail_reasons") or [])]


def category(d: str) -> str:
    dl = d.lower()
    if "tamper" in dl or "frozen" in dl or "writable" in dl or "pythonpath" in dl:
        return "forbidden_edit_path"
    if ("parity" in dl or "rel_err" in dl or "exact" in dl or "poison" in dl
            or "alias" in dl or "quality" in dl or "gate" in dl):
        return "correctness_failed"
    if "entry contract" in dl or "missing" in dl or "not found" in dl:
        return "correctness_failed"
    if "plausib" in dl or "bound" in dl or "budget" in dl or "timeout" in dl:
        return "timing_invalid"
    return "verifier_completed"


# 🔴 reward.md gate 5 (`speedup <= 1`) is a LEGITIMATE ZERO, not a candidate defect: the run
#    completed and EVERY correctness case passed, the candidate just did not beat the strong
#    baseline. Listing it in `hard_fails` makes /opt/loop's score_engine report
#    "correctness: FAIL / the check did not complete", which is false and would be the default
#    per-round feedback for most of a loop16 trajectory. The reward stays 0.0 either way; every
#    REAL hard-fail reason (correctness / tamper / cheating / entry contract) still surfaces.
BENIGN_ZERO = {"speedup_not_above_baseline"}
real_fails = [x for x in spec_fails if x not in BENIGN_ZERO]
hard = [] if (ok and not real_fails) else (
    [category(detail) if LOOP else (real_fails[0] if real_fails
                                    else (spec_fails[0] if spec_fails else detail[:220]))])
print(json.dumps({"gates": {"correctness_ok": ok}, "hard_fails": hard,
                  "reward": rew, "speedup": sp}))

if LOOP:
    red = {"reward": rew, "quality_gate_passed": ok,
           "failing_category": (hard[0] if hard else None),
           "note": "REDACTED for the per-round development loop: case identities, tolerances, "
                   "per-case measurements and hardware constants are part of the evaluation "
                   "surface and are not disclosed. The end-of-session score is authoritative."}
    for p in glob.glob('/logs/verifier/*.json'):
        try:
            json.dump(red, open(p, 'w'))
        except Exception:
            pass
    # the saved candidate artifacts + any raw per-case dump also carry measurement detail; drop them.
    for p in (glob.glob('/logs/verifier/*.txt') + glob.glob('/logs/verifier/*.log')
              + glob.glob('/logs/verifier/.*.log')):
        if p.endswith('reward.txt'):
            continue
        try:
            os.remove(p)
        except Exception:
            pass
PYEOF
exit 0
