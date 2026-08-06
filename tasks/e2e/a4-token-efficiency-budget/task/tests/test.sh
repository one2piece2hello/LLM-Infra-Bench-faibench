#!/bin/bash
# e2e-a4-token-efficiency-budget verifier entry (family A, single-shot, train-and-eval scoring).
#
# The verifier RUNS the candidate's train_for_budget recipe once per seed under a HARNESS-OWNED
# **TOKEN** budget -- materialised as a per-seed shard of exactly token_budget tokens with the
# full corpus chmod-000 quarantined, so the budget holds BY CONSTRUCTION and is never a number
# the candidate reports -- plus a generous wall-clock FEASIBILITY GUARD enforced by an external
# process-group kill. It then reloads each checkpoint in a clean process, re-counts parameters
# against the cap, requires distinct weights across seeds, runs the anti-spoof kit, and scores
# the MEDIAN held-out bpb.
#
# REWARD (the bench reward spec, performance class, BOUNDED):
#     speedup     = baseline_bpb / median_val_bpb          (quality ratio at a FIXED token budget)
#     ref_speedup = baseline_bpb / oracle_val_bpb          (FROZEN authoring constant)
#     reward      = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0            in [0, 1]
# Matching the tuned-AdamW recipe the solver started from scores 0 (pre-gate 5); matching the
# demonstrated in-budget ceiling scores 0.5. The scorer reads the two constants from the frozen
# manifest and NEVER runs the oracle or the baseline.
#
# The frozen surface (verifier + holdout + manifest) is uploaded FRESH under /tests at scoring.
# 🔴 There is no baked /opt/verifier fallback any more: it leaked the held-out corpus and the
# calibrated anchor into the solver's own container (uid 0 => root-0700 is no protection).

set -o pipefail

# --- step 0: PATH pin (the harness may exec login-style, the agent shell is non-login; pin the venv
#     so python AND the training subprocesses the verifier spawns have torch). ---
export PATH="/opt/kernelbench-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PY="$(command -v python3 || command -v python)"
# Pin BLAS/OMP thread counts so runtime does not depend on node topology (measured on this
# fleet: a 145-core node let torch open 96 threads and ran 4.3x slower than a pinned host).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

SUBMISSION_DIR="${SUBMISSION_DIR:-/app/submission}"
FROZEN=/tests
VDIR=/logs/verifier
mkdir -p "$VDIR" /logs/artifacts "${SUBMISSION_DIR}"
rm -f "$VDIR"/reward.txt "$VDIR"/reward.json "$VDIR"/metrics.json \
      "$VDIR"/verifier_state.json "$VDIR"/correctness_results.json "$VDIR"/benchmark_results.json
cd /            # a deleted CWD kills torch's dlopen; anchor somewhere that always exists

# The complete verifier output contract on EVERY hard-fail path, delegated to the frozen
# scorer's --emit-zero so the hard-fail schema can never drift from the success schema. The
# inline fallback covers the case where python itself cannot start.
emit_zero() {  # $1 = human message, $2 = named reason
    _r="${2:-build_or_entry_contract_failed}"
    _msg=$(printf '%s' "$1" | tr -d '"\\')
    echo "HARD-FAIL[$_r]: $1"
    if [ -f "$FROZEN/compute_reward.py" ] && "$PY" "$FROZEN/compute_reward.py" \
          --emit-zero "$_r" --detail "$_msg" >/dev/null 2>&1; then
        return 0
    fi
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"partial_score\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"passed\":0,\"total\":0,\"tests\":{\"passed\":0,\"total\":0},\"metric_kind\":\"quality_at_fixed_budget\",\"metric_name\":\"val_bpb\",\"metric_direction\":\"lower_is_better\",\"timing_measured\":false,\"error\":\"${_msg}\"}" > "$VDIR/metrics.json"
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"cv\":{},\"metric_kind\":\"quality_at_fixed_budget\",\"metric_name\":\"val_bpb\",\"metric_direction\":\"lower_is_better\",\"timing_measured\":false}" > "$VDIR/reward.json"
    echo "{\"task_id\":\"e2e-a4-token-efficiency-budget\",\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"detail\":\"${_msg}\"}" > "$VDIR/verifier_state.json"
    echo "{\"passed\":0,\"total\":0,\"all_passed\":false,\"hard_fail_reasons\":[\"${_r}\"],\"failed_checks\":[{\"category\":\"entry_contract\",\"name\":\"${_r}\",\"message\":\"${_msg}\"}]}" > "$VDIR/correctness_results.json"
    echo "{\"metric_kind\":\"quality_at_fixed_budget\",\"metric_name\":\"val_bpb\",\"metric_direction\":\"lower_is_better\",\"timing_measured\":false,\"candidate_val_bpb\":null,\"speedup\":null,\"metric_void_on_hard_fail\":true,\"oracle_executed_by_scorer\":false,\"baseline_executed_by_scorer\":false,\"hard_fail_reasons\":[\"${_r}\"]}" > "$VDIR/benchmark_results.json"
    echo 0.000000 > "$VDIR/reward.txt"
}

save_artifacts() {
    for f in train_gpt.py verification_report.md action.log train.log; do
        [ -e "${SUBMISSION_DIR}/${f}" ] && cp -r "${SUBMISSION_DIR}/${f}" "/logs/artifacts/${f}" 2>/dev/null || true
    done
}
trap save_artifacts EXIT

# --- step 1: 🔴 NO MODE DISPATCH. The previous version copied a reviewer recipe into
#     /app/submission when VERIFIER_MODE was set -- and VERIFIER_MODE came from the
#     environment, which the solver owns (the entrypoint is a LOGIN shell that sources
#     ~/.bashrc). Combined with the baked /opt/strong_baseline that was a harness backdoor
#     into the graded tree. Reviewer modes are now driven by the AUTHOR's calibration driver,
#     which stages recipes itself; the scored path grades /app/submission and nothing else. ---
if [ -n "${VERIFIER_MODE:-}" ] && [ "${VERIFIER_MODE}" != "candidate" ]; then
    emit_zero "VERIFIER_MODE=${VERIFIER_MODE} is not honoured on the scored path (harness mode dispatch removed)" "anti_cheat_triggered"
    exit 1
fi

# --- step 2: the frozen surface must be the FRESH /tests upload, and ONLY that. -----------
FROZEN_SET="compute_reward.py test.sh holdout_corpus.txt verifier-correctness-manifest.json"
if [ ! -f "$FROZEN/compute_reward.py" ]; then
    if [ -e /opt/verifier ]; then
        emit_zero "frozen surface not mounted at /tests and an in-image /opt/verifier copy is present — refusing to score against a solver-reachable evaluation surface" "forbidden_edit_path"
        exit 1
    fi
    emit_zero "frozen evaluation surface missing: /tests/compute_reward.py not mounted — failing closed" "build_or_entry_contract_failed"
    exit 1
fi
for f in $FROZEN_SET; do
    [ -f "$FROZEN/$f" ] || { emit_zero "frozen evaluation surface incomplete: /tests/$f not mounted" "build_or_entry_contract_failed"; exit 1; }
done
for p in /opt/verifier /opt/strong_baseline /opt/naive /opt/negative /opt/ceiling; do
    if [ -e "$p" ]; then
        emit_zero "in-image reference/evaluation surface present at $p (image is stale / re-leaked)" "forbidden_edit_path"
        exit 1
    fi
done

MANIFEST="$FROZEN/verifier-correctness-manifest.json"
exp_sha=$("${PY}" -c "import json;print(json.load(open('${MANIFEST}')).get('compute_reward_sha256',''))" 2>/dev/null)
got_sha=$(sha256sum "$FROZEN/compute_reward.py" | awk '{print $1}')
if [ -n "${exp_sha}" ] && [ "${exp_sha}" != "${got_sha}" ]; then
    emit_zero "verifier sha256 mismatch (frozen surface tampered): expected ${exp_sha} got ${got_sha}" "forbidden_edit_path"
    exit 1
fi

# --- step 3: required-file presence (entry contract). ---
if [ ! -f "${SUBMISSION_DIR}/train_gpt.py" ]; then
    emit_zero "missing ${SUBMISSION_DIR}/train_gpt.py (needs train_for_budget + load_model_for_verification)" "build_or_entry_contract_failed"
    exit 1
fi

# --- step 4: run the scorer. It trains the recipe N_SEEDS times under the token budget, evals
#     each checkpoint (quality gate + PARAM CAP + anti-spoof + cross-seed divergence), takes the
#     median bpb, and applies the bounded reward. It is NOT run with `-I` because it must spawn
#     training subprocesses that import torch/nanoGPT from the venv; sanitize_python_path (inside
#     the scorer) + PYTHONSAFEPATH stop the submission module from shadowing it.
#
#     Every E2E_*_OVERRIDE is stripped: on the scored path no environment variable may influence
#     the budget, the seed set or the frozen reward constants. (The scorer no longer reads them
#     at all; this is belt-and-braces for anything downstream.) ---
for v in $(env | sed -n 's/^\(E2E_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset PYTHONPATH PYTHONHOME VERIFIER_MODE
STAGE=$(mktemp -d /tmp/.e2e_a4_stage.XXXXXXXX); chmod 0700 "$STAGE"
cp "$FROZEN/compute_reward.py" "$STAGE/compute_reward.py"
SUBMISSION_DIR="${SUBMISSION_DIR}" PYTHONSAFEPATH=1 "${PY}" "$STAGE/compute_reward.py"
verify_exit=$?

# --- step 5: output-contract guarantee. The scorer writes all six itself on every path it
#     reaches (its own crash handler included), but if the interpreter dies first (missing dep /
#     OOM / SIGKILL) nothing would be written at all. ---
CONTRACT_MISSING=false
for f in reward.txt reward.json metrics.json verifier_state.json correctness_results.json benchmark_results.json; do
    [ -s "$VDIR/$f" ] || CONTRACT_MISSING=true
done
if [ "$CONTRACT_MISSING" = true ]; then
    emit_zero "verifier process died before writing the result contract (exit ${verify_exit})" "verifier_crashed"
    rm -rf "$STAGE"
    exit 1
fi
echo "[test] compute_reward rc=${verify_exit} ; reward=$(cat "$VDIR/reward.txt" 2>/dev/null)"
rm -rf "$STAGE"
exit "${verify_exit}"
