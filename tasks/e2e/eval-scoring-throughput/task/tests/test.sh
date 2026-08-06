#!/bin/bash
# e2e-h3-eval-harness-throughput-quality verifier entry (family C, single-shot, eval-only).
# 7-step canonical structure adapted for an eval-harness scoring-throughput-under-consistency
# task. The frozen surface (verifier + held-out sample set + manifest) is uploaded FRESH under
# /tests at scoring and exists NOWHERE in the task image (2026-07-27: the baked root-0700
# /opt/verifier fallback was removed -- the container runs as uid 0, so root-0700 protected nothing).
# Nothing the submission reports is trusted: the harness owns the sample set, the gold targets,
# an INDEPENDENT reference scorer, and the clock; it re-times real work and requires EXACT
# per-sample agreement.

set -o pipefail

# --- step 0: PATH pin (login-style exec is possible; pin the venv so `python` has torch). ---
export PATH="/opt/kernelbench-venv/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
PY="$(command -v python3 || command -v python)"

SUBMISSION_DIR="${SUBMISSION_DIR:-/app/submission}"
MODE="${VERIFIER_MODE:-candidate}"     # candidate | strong_baseline | negative
mkdir -p /logs/verifier /logs/artifacts "${SUBMISSION_DIR}"
rm -f /logs/verifier/reward.txt /logs/verifier/metrics.json /logs/verifier/reward.json

emit_zero() {
    _r="${2:-build_or_entry_contract_failed}"
    echo "$1"
    # the FULL 5-file verifier contract on every hard-fail path (the pre-2026-07-27 version
    # wrote only 3 files, so a tamper/entry-contract fail left the contract incomplete).
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"partial_score\":0.0,\"binary_pass\":0,\"hard_fail_reasons\":[\"${_r}\"],\"error\":\"$1\"}" > /logs/verifier/metrics.json
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"cv\":{},\"binary_pass\":0}" > /logs/verifier/reward.json
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"detail\":\"$1\"}" > /logs/verifier/verifier_state.json
    echo "{\"binary_pass\":0,\"all_hard_gates_pass\":false,\"detail\":\"$1\"}" > /logs/verifier/correctness_results.json
    echo "{\"score_metric\":\"log_speedup_vs_ref_speedup\",\"reward\":0.0,\"speedup\":null,\"ref_speedup\":null,\"metric_void_on_hard_fail\":true,\"detail\":\"$1\"}" > /logs/verifier/benchmark_results.json
    echo 0.000000 > /logs/verifier/reward.txt
}

save_artifacts() {
    for f in scoring_pipeline.py scoring_config.json verification_report.md action.log; do
        [ -e "${SUBMISSION_DIR}/${f}" ] && cp -r "${SUBMISSION_DIR}/${f}" "/logs/artifacts/${f}" 2>/dev/null || true
    done
}
trap save_artifacts EXIT

# --- step 1: prepare clean candidate state. MODE dispatch: the reviewer-only
#     strong_baseline / negative / ceiling seeding NEVER runs in candidate mode.
#     🔴 /opt/strong_baseline and /opt/negative are NOT baked into the image (MOD_SPEC 改动 3);
#     a reviewer validating a non-candidate mode uploads the seed dir into the live session and
#     points STRONG_BASELINE_DIR / NEGATIVE_DIR at it. ---
if [ "${MODE}" != "candidate" ]; then
    case "${MODE}" in
        strong_baseline)
            SEED_DIR="${STRONG_BASELINE_DIR:-/opt/strong_baseline}"
            cp "${SEED_DIR}/scoring_pipeline.py" "${SUBMISSION_DIR}/scoring_pipeline.py"
            [ -f "${SEED_DIR}/scoring_config.json" ] && cp "${SEED_DIR}/scoring_config.json" "${SUBMISSION_DIR}/scoring_config.json"
            echo "[mode] seeded STRONG BASELINE into ${SUBMISSION_DIR}" ;;
        negative|ceiling)
            SEED_DIR="${NEGATIVE_DIR:-/opt/negative}"
            cp "${SEED_DIR}/scoring_pipeline.py" "${SUBMISSION_DIR}/scoring_pipeline.py"
            echo "[mode] seeded ${MODE} into ${SUBMISSION_DIR}" ;;
        *) echo "[mode] unknown VERIFIER_MODE=${MODE}, treating as candidate" ;;
    esac
fi

# --- step 2: frozen-surface hard gate. The frozen surface is ONLY the FRESH /tests upload.
#     🔴 2026-07-27: the baked /opt/verifier fallback (stale unbounded scorer + the held-out
#     sample set + the calibration manifest) has been REMOVED from the image and is no longer
#     consulted here, so a missing upload fails CLOSED instead of scoring with stale semantics
#     against a solver-readable sample set. self-check sha256 vs the manifest. ---
VERIFIER="/tests/compute_reward.py"
if [ ! -f "${VERIFIER}" ]; then emit_zero "verifier missing (expected the fresh /tests upload)"; exit 1; fi

MANIFEST="/tests/verifier-correctness-manifest.json"
if [ ! -f "${MANIFEST}" ]; then emit_zero "frozen manifest missing (expected the fresh /tests upload)"; exit 1; fi
if [ ! -f "/tests/heldout_samples.jsonl" ]; then emit_zero "held-out sample set missing (expected the fresh /tests upload)"; exit 1; fi
exp_sha=$("${PY}" -c "import json;print(json.load(open('${MANIFEST}')).get('compute_reward_sha256',''))" 2>/dev/null)
got_sha=$(sha256sum "${VERIFIER}" | awk '{print $1}')
if [ -n "${exp_sha}" ] && [ "${exp_sha}" != "${got_sha}" ]; then
    emit_zero "verifier sha256 mismatch (frozen surface tampered)" "forbidden_edit_path"; exit 1
fi

# --- step 3: required-file presence (entry contract). ---
if [ ! -f "${SUBMISSION_DIR}/scoring_pipeline.py" ]; then emit_zero "missing scoring_pipeline.py"; exit 1; fi

# --- step 3b: source scan (G-gate). Scans ONLY the solver-editable submission surface:
#     widening this to /app would match the frozen /tests copy itself and zero every mode.
#     Runs in every MODE, so the reference solution must clear the same bar a candidate does. ---
if [ -e "${SUBMISSION_DIR}/compute_reward.py" ] || [ -e "${SUBMISSION_DIR}/verifier-correctness-manifest.json" ]; then
    emit_zero "candidate shadowed a frozen filename inside ${SUBMISSION_DIR}" "forbidden_edit_path"; exit 1
fi
SUSPICIOUS="/tests/|/opt/verifier|/opt/strong_baseline|/opt/negative|compute_reward|verifier-correctness-manifest|/logs/verifier|reward\\.json|reward\\.txt"
SCAN_FAILED=""
while IFS= read -r -d '' f; do
    if grep -qE "${SUSPICIOUS}" "$f" 2>/dev/null; then SCAN_FAILED="$f"; break; fi
done < <(find "${SUBMISSION_DIR}" -type f \
    \( -name "*.py" -o -name "*.sh" -o -name "*.json" -o -name "*.toml" \
       -o -name "*.yaml" -o -name "*.yml" -o -name "*.cfg" -o -name "*.ini" \) \
    -not -path "*/.*" -print0 2>/dev/null)
if [ -n "${SCAN_FAILED}" ]; then emit_zero "source references verifier internals: ${SCAN_FAILED}" "cheating_verifier_reference"; exit 1; fi
echo "PASS: source scan"

# --- steps 4-7: consistency gate + anti-cache probe + harness-owned ABBA timing (>=5 alternating
#     baseline/candidate pairs, the baseline RE-MEASURED in-session from
#     /tests/oracles/strong_baseline_scoring_pipeline.py) + the bounded log-speedup reward
#     (reward.md) are all computed by the eval-only verifier in one isolated pass. `-I` +
#     unset PYTHONPATH/PYTHONHOME + PYTHONSAFEPATH=1 stop the submission module from shadowing
#     the verifier (sanitize_python_path also runs).
#     🔴 2026-07-28 (vacuity sweep V7): `-I` and sanitize_python_path only stop SHADOWING. The
#     candidate is still loaded with importlib INTO this process, so it shares `time`, the
#     held-out workload, the harness's own reference scorer and the reward function with the
#     measurement. MEASURED bypasses (all with hard_fail_reasons=[]): a byte-identical copy of
#     the strong baseline + a selective idempotent clock-debt patch -> reward 0.9477 at speedup
#     4.6283; `compute_log_reward = lambda: 1.0` -> reward 1.0 on a candidate SLOWER than the
#     baseline and on the sample-skipping WRONG negative; `_reference_score_one = lambda s: 0.0`
#     -> reward 1.0 at speedup 15.0002 for a scorer that reads no field at all. compute_reward.py
#     now binds every such primitive before the candidate import and hard-fails a replacement
#     with the named reason `cheating_harness_tamper`. ---
cp "${VERIFIER}" /tmp/e2e_eval_throughput_verifier.py
unset PYTHONPATH PYTHONHOME
SUBMISSION_DIR="${SUBMISSION_DIR}" PYTHONSAFEPATH=1 "${PY}" -I /tmp/e2e_eval_throughput_verifier.py
verify_exit=$?

# --- FINAL CONTRACT GUARD: a hard interpreter abort (OOM / segfault) can leave /logs/verifier
#     empty even though the verifier "ran". A missing contract must never surface as a silent
#     no-result -> synthesise the full contract with reward 0.0 and a NAMED reason. ---
for f in reward.txt reward.json metrics.json benchmark_results.json correctness_results.json verifier_state.json; do
    if [ ! -s "/logs/verifier/${f}" ]; then
        emit_zero "the verifier did not produce the full result contract (missing ${f})" "verifier_crashed_without_contract"
        break
    fi
done

if [ "${verify_exit}" -ne 0 ]; then
    exit "${verify_exit}"
fi
exit 0
