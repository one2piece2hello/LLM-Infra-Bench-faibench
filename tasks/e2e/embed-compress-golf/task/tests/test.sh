#!/bin/bash
# e2e-g2-embed-compress-golf verifier entry (family C, single-shot, eval-only).
# 7-step canonical structure adapted for an eval-only embedding-quality-under-budget
# task. The frozen surface (verifier + held-out corpus/queries/qrels + manifest) lives
# root-0700 at /opt/verifier as a baked fallback AND is uploaded FRESH under /tests at
# scoring (solver-invisible). Nothing the submission reports is trusted: the harness
# encodes OUR held-out text, re-measures the byte budget, runs OUR retrieval + metric.

set -o pipefail

# --- step 0: PATH pin (login-style exec is possible; pin the venv so `python` has torch/st). ---
export PATH="/opt/kernelbench-venv/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
PY="$(command -v python3 || command -v python)"

# --- step 0b: DETERMINISTIC SCORING ENVIRONMENT (2026-07-27) -----------------------------------
# (a) THREAD PINNING. Measured on the CPU lane: the node exposes 145 cores, torch therefore
#     defaults to 96 threads, and a 200-document encode takes 10.7 s; pinned to 8 it takes 2.5 s
#     -- a 4.3x swing driven purely by the node's core count. A scored run whose wall time moves
#     4.3x with node topology is a latent timeout flake, so the thread count is pinned to the
#     declared [environment].cpus instead of being inherited from the host.
#     This cannot move the score: nDCG@10 is exact arithmetic over a ranking, and 8 threads is the
#     setting under which the frozen constants below were reproduced bit-for-bit.
# (b) DEVICE PINNING. strong_baseline_ndcg = 0.459151 and ref_speedup = 1.4290238072817099 were
#     calibrated on CPU. task.toml grants the SOLVER an H20 (encoder iteration / optional
#     fine-tuning of the base model), but compute_reward.py picks its device with
#     `cuda if torch.cuda.is_available() else cpu`, so on a GPU node scoring would silently move to
#     CUDA, where a different float reduction order can flip near-ties in the top-50 shortlist and
#     shift nDCG away from the split the constants were measured on. Scoring is therefore pinned to
#     CPU regardless of the card present. Solver-side GPU use is unaffected (this only wraps the
#     verifier process).
E2E_SCORING_THREADS="${E2E_SCORING_THREADS:-8}"     # == task.toml [environment].cpus
export OMP_NUM_THREADS="${E2E_SCORING_THREADS}" \
       MKL_NUM_THREADS="${E2E_SCORING_THREADS}" \
       OPENBLAS_NUM_THREADS="${E2E_SCORING_THREADS}" \
       NUMEXPR_NUM_THREADS="${E2E_SCORING_THREADS}" \
       TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
echo "[env] scoring pinned: threads=${E2E_SCORING_THREADS} CUDA_VISIBLE_DEVICES='' (CPU-only, matches the frozen calibration)"

SUBMISSION_DIR="${SUBMISSION_DIR:-/app/submission}"
MODE="${VERIFIER_MODE:-candidate}"     # candidate | strong_baseline | negative
mkdir -p /logs/verifier /logs/artifacts "${SUBMISSION_DIR}"
rm -f /logs/verifier/reward.txt /logs/verifier/metrics.json /logs/verifier/reward.json

emit_zero() {
    _r="${2:-build_or_entry_contract_failed}"
    echo "$1"
    # the FULL 5-file verifier contract on every hard-fail path, in the reward.md result shape
    # (before 2026-07-27 this wrote 3 files with a bare {"reward":0,"binary_pass":0}).
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"partial_score\":0.0,\"binary_pass\":0,\"hard_fail_reasons\":[\"${_r}\"],\"error\":\"$1\"}" > /logs/verifier/metrics.json
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"cv\":{},\"metric_kind\":\"quality_ratio\",\"metric_name\":\"ndcg@10\",\"metric_direction\":\"higher_is_better\",\"timing_measured\":false,\"binary_pass\":0}" > /logs/verifier/reward.json
    echo "{\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"detail\":\"$1\"}" > /logs/verifier/verifier_state.json
    echo "{\"binary_pass\":0,\"all_hard_gates_pass\":false,\"detail\":\"$1\"}" > /logs/verifier/correctness_results.json
    echo "{\"score_metric\":\"ndcg\",\"reward\":0.0,\"quality_ratio_vs_strong_baseline\":null,\"metric_void_on_hard_fail\":true,\"detail\":\"$1\"}" > /logs/verifier/benchmark_results.json
    echo 0.000000 > /logs/verifier/reward.txt
}

save_artifacts() {
    for f in submission_encoder.py encoder_config.json verification_report.md action.log; do
        [ -e "${SUBMISSION_DIR}/${f}" ] && cp -r "${SUBMISSION_DIR}/${f}" "/logs/artifacts/${f}" 2>/dev/null || true
    done
}
trap save_artifacts EXIT

# --- step 1: prepare clean candidate state. MODE dispatch: the reviewer-only
#     strong_baseline / negative seeding NEVER runs in candidate mode. ---
if [ "${MODE}" != "candidate" ]; then
    case "${MODE}" in
        strong_baseline)
            SEED_DIR="${STRONG_BASELINE_DIR:-/opt/strong_baseline}"
            cp "${SEED_DIR}/submission_encoder.py" "${SUBMISSION_DIR}/submission_encoder.py"
            [ -f "${SEED_DIR}/encoder_config.json" ] && cp "${SEED_DIR}/encoder_config.json" "${SUBMISSION_DIR}/encoder_config.json"
            echo "[mode] seeded STRONG BASELINE into ${SUBMISSION_DIR}" ;;
        negative)
            SEED_DIR="${NEGATIVE_DIR:-/opt/negative}"
            cp "${SEED_DIR}/submission_encoder.py" "${SUBMISSION_DIR}/submission_encoder.py"
            echo "[mode] seeded NEGATIVE into ${SUBMISSION_DIR}" ;;
        *) echo "[mode] unknown VERIFIER_MODE=${MODE}, treating as candidate" ;;
    esac
fi

# --- step 2: frozen-surface hard gate. Prefer the FRESH /tests verifier (never a
#     baked copy the solver may have shadowed); self-check its sha256 vs the manifest. ---
VERIFIER="/tests/compute_reward.py"
[ -f "${VERIFIER}" ] || VERIFIER="/opt/verifier/compute_reward.py"
if [ ! -f "${VERIFIER}" ]; then emit_zero "verifier missing"; exit 1; fi

MANIFEST="/tests/verifier-correctness-manifest.json"
[ -f "${MANIFEST}" ] || MANIFEST="/opt/verifier/verifier-correctness-manifest.json"
if [ -f "${MANIFEST}" ]; then
    exp_sha=$("${PY}" -c "import json;print(json.load(open('${MANIFEST}')).get('compute_reward_sha256',''))" 2>/dev/null)
    got_sha=$(sha256sum "${VERIFIER}" | awk '{print $1}')
    if [ -n "${exp_sha}" ] && [ "${exp_sha}" != "${got_sha}" ]; then
        emit_zero "verifier sha256 mismatch (frozen surface tampered)" "forbidden_edit_path"; exit 1
    fi
fi

# --- step 3: required-file presence (entry contract). ---
if [ ! -f "${SUBMISSION_DIR}/submission_encoder.py" ]; then emit_zero "missing submission_encoder.py"; exit 1; fi

# --- steps 4-7: budget gate + anti-degenerate probes + harness-owned retrieval +
#     held-out nDCG + the BOUNDED [0,1] reward.md log reward are all computed by the eval-only
#     verifier in one isolated pass. `-I` + unset PYTHONPATH/PYTHONHOME +
#     PYTHONSAFEPATH=1 stop the submission module from shadowing the verifier
#     (sanitize_python_path also runs). ---
cp "${VERIFIER}" /tmp/e2e_embed_compress_verifier.py
unset PYTHONPATH PYTHONHOME
SUBMISSION_DIR="${SUBMISSION_DIR}" PYTHONSAFEPATH=1 "${PY}" -I /tmp/e2e_embed_compress_verifier.py
verify_exit=$?

# --- 5-FILE CONTRACT GUARANTEE (2026-07-27): the verifier writes all five itself on every path it
#     reaches (including its own internal exception handler), but if the interpreter dies before
#     that (missing dep, OOM, SIGKILL) nothing would be written at all. Backfill the full contract
#     with reward 0.0 + a named reason so a hard-fail path can never emit fewer than five files. ---
_contract_missing=false
for f in reward.txt reward.json metrics.json verifier_state.json correctness_results.json benchmark_results.json; do
    [ -f "/logs/verifier/${f}" ] || _contract_missing=true
done
if [ "${_contract_missing}" = true ]; then
    emit_zero "verifier process died before writing the result contract (exit ${verify_exit})" "verifier_crashed"
    exit 1
fi

if [ "${verify_exit}" -ne 0 ]; then
    exit "${verify_exit}"
fi
exit 0
