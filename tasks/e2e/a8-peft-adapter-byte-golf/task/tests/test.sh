#!/bin/bash
# e2e-a8-peft-adapter-byte-golf verifier entry (family C, single-shot, EVAL-ONLY).
#
# The solver iterates its own PEFT recipe inside its agent time budget and submits TWO
# byte-capped files; this harness evals them ONCE. It NEVER re-runs the solver's
# fine-tune (the strong_baseline / ceiling / naive MODES do run a reviewer-only reference
# recipe — that is the authoring-time calibration path, never the candidate path).
#
# 7-step canonical structure for an eval-only
# quality-under-an-ADAPTER-BYTE-budget task. reward 0 on ANY hard-fail (missing artifact
# / byte budget exceeded / base weights tampered / anti-spoof fail / build hook crash).
#
# 🔴 reward is reward.md 性能类 BOUNDED to [0,1] —
#    min(1.0, ln(gain_ratio/ref_speedup)/ln(ref_speedup)) if gain_ratio > ref_speedup else 0.0 — NOT the old un-capped open-ended ratio.
#    Every hard-fail path now emits the COMPLETE 6-file contract with a NAMED reason (it used
#    to emit 3 files and a bare {"reward":0.0,"binary_pass":0}).
# 🔴 The frozen surface and every reviewer-only recipe are NO
#    LONGER baked into the image. /tests is uploaded fresh at scoring; a reviewer validating a
#    non-candidate MODE uploads task/solution/* into the live session and points
#    STRONG_BASELINE_DIR / CEILING_DIR / NEGATIVE_DIR / STASH_ORACLE_DIR at the upload dir.
#
# dash has no `set -o pipefail` (the container shell may be dash) -> guard it
(set -o pipefail) 2>/dev/null && set -o pipefail || true

# --- step 0: PATH pin. The harness may exec login-style but the agent shell is NON-login and
#     the base is a venv image, so /opt/kernelbench-venv/bin MUST lead PATH or a bare
#     python3 resolves to a torch-less system python (the shared family-A test.sh bug). ---
export PATH=/opt/kernelbench-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PY="$(command -v python3 || command -v python)"
# The image WORKDIR is /app/submission, which the verifier's own quarantine step renames out of
# the way. A shell whose CWD inode has been unlinked makes getcwd() fail and torch/MKL then dies
# with "Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so". Anchor somewhere that cannot move.
cd / || true

SUBMISSION_DIR="${SUBMISSION_DIR:-/app/submission}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/app/base_model}"
MODE="${VERIFIER_MODE:-candidate}"   # candidate | strong_baseline | ceiling | naive | negative_overbudget | negative_degenerate | negative_stash
mkdir -p /logs/verifier /logs/artifacts "${SUBMISSION_DIR}"
rm -f /logs/verifier/reward.txt /logs/verifier/metrics.json /logs/verifier/reward.json \
      /logs/verifier/correctness_results.json /logs/verifier/benchmark_results.json \
      /logs/verifier/verifier_state.json

# 🔴 the hard-fail path MUST emit the COMPLETE verifier contract (metrics.json / reward.json /
#    reward.txt / correctness_results.json / benchmark_results.json / verifier_state.json) with a
#    NAMED reward.md 前置门 reason, in the reward.md 性能类 shape. Called from EVERY short-circuit.
emit_zero() {
    _r="${2:-build_or_entry_contract_failed}"
    echo "HARD-FAIL: $1"
    _msg=$(printf '%s' "$1" | tr -d '"\\')
    _core="\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"cv\":{\"baseline\":0.0,\"candidate\":0.0},\"metric_kind\":\"quality_ratio_NOT_time_speedup\",\"metric_name\":\"adaptation_gain_ratio\",\"metric_direction\":\"higher_is_better\",\"timing_measured\":false"
    echo "{${_core},\"partial_score\":0.0,\"binary_pass\":0,\"quality_gate_passed\":false,\"tests\":{\"passed\":0,\"total\":0},\"error\":\"${_msg}\"}" > /logs/verifier/metrics.json
    echo "{${_core}}" > /logs/verifier/reward.json
    echo 0.000000 > /logs/verifier/reward.txt
    echo "{\"checks\":[{\"category\":\"entry_contract\",\"name\":\"${_r}\",\"passed\":false,\"message\":\"${_msg}\",\"hard\":true}],\"passed\":0,\"total\":0,\"all_hard_gates_pass\":false,\"all_gates_pass\":false,\"hard_fail_reasons\":[\"${_r}\"],\"failed_checks\":[{\"category\":\"entry_contract\",\"name\":\"${_r}\",\"message\":\"${_msg}\",\"hard\":true}]}" > /logs/verifier/correctness_results.json
    echo "{\"score_metric\":\"heldout_ce\",\"base_ce\":null,\"candidate_ce\":null,\"gain_ratio_vs_strong_baseline\":null,\"ref_speedup\":null,\"metric_kind\":\"quality_ratio_NOT_time_speedup\",\"timing_measured\":false,\"metric_void_on_hard_fail\":true,\"hard_fail_reasons\":[\"${_r}\"]}" > /logs/verifier/benchmark_results.json
    echo "{\"task_id\":\"e2e-a8-peft-adapter-byte-golf\",\"task_type\":\"performance\",\"mode\":\"${MODE}\",\"reward\":0.0,\"all_hard_pass\":false,\"hard_fail_reasons\":[\"${_r}\"],\"speedup\":null,\"ref_speedup\":null,\"detail\":\"${_msg}\"}" > /logs/verifier/verifier_state.json
}

save_artifacts() {
    for f in adapter_entry.py train_adapter.py train.log verification_report.md; do
        [ -e "${SUBMISSION_DIR}/${f}" ] && cp -r "${SUBMISSION_DIR}/${f}" "/logs/artifacts/${f}" 2>/dev/null || true
    done
    [ -e "${SUBMISSION_DIR}/adapter.bin" ] && ls -l "${SUBMISSION_DIR}/adapter.bin" > /logs/artifacts/adapter_bin.ls 2>/dev/null || true
}
trap save_artifacts EXIT

# --- step 1: MODE dispatch. The reviewer-only reference/negative seeding NEVER runs in
#     candidate mode (a re-seeded test.sh must not silently clobber candidate mode). Validation drives VERIFIER_MODE.
#     Nothing is baked any more, so every non-candidate mode FAILS CLOSED with a named reason
#     if its recipe was not uploaded. ---
if [ "${MODE}" != "candidate" ]; then
    case "${MODE}" in
        strong_baseline)
            SEED_DIR="${STRONG_BASELINE_DIR:-/opt/strong_baseline}"
            if [ ! -f "${SEED_DIR}/train_adapter.py" ]; then
                emit_zero "strong-baseline seed dir ${SEED_DIR} is empty (upload task/solution/strong_baseline and set STRONG_BASELINE_DIR)" "reviewer_seed_missing"; exit 1
            fi
            cp "${SEED_DIR}/adapter_entry.py" "${SUBMISSION_DIR}/adapter_entry.py"
            echo "[mode] TRAINING the reviewer-only STRONG reference recipe (calibration path)"
            ( cd "${SUBMISSION_DIR}" && A8_TRAIN_LOG=/logs/artifacts/strong_train.log \
              "${PY}" "${SEED_DIR}/train_adapter.py" 2>&1 | tail -40 ) || { emit_zero "strong-baseline training failed" "reviewer_seed_failed"; exit 1; }
            ;;
        ceiling)
            # the reviewer-only IN-BUDGET CEILING recipe: the ORACLE whose MEASURED gain ratio IS
            # the frozen ref_speedup, so this mode is the one that must score ~0.5.
            SEED_DIR="${CEILING_DIR:-/opt/ceiling}"
            if [ ! -f "${SEED_DIR}/train_adapter.py" ]; then
                emit_zero "ceiling seed dir ${SEED_DIR} is empty (upload task/solution/ceiling and set CEILING_DIR)" "reviewer_seed_missing"; exit 1
            fi
            cp "${SEED_DIR}/adapter_entry.py" "${SUBMISSION_DIR}/adapter_entry.py"
            echo "[mode] TRAINING the reviewer-only IN-BUDGET CEILING recipe (the ref_speedup oracle)"
            ( cd "${SUBMISSION_DIR}" && A8_TRAIN_LOG=/logs/artifacts/ceiling_train.log \
              "${PY}" "${SEED_DIR}/train_adapter.py" 2>&1 | tail -40 ) || { emit_zero "ceiling training failed" "reviewer_seed_failed"; exit 1; }
            ;;
        naive)
            echo "[mode] TRAINING the model-visible NAIVE starter recipe"
            ( cd "${SUBMISSION_DIR}" && "${PY}" "${SUBMISSION_DIR}/train_adapter.py" 2>&1 | tail -40 ) \
              || { emit_zero "naive training failed" "reviewer_seed_failed"; exit 1; }
            ;;
        negative_overbudget)
            _sd="${STRONG_BASELINE_DIR:-/opt/strong_baseline}"
            [ -f "${_sd}/adapter_entry.py" ] || { emit_zero "over-budget negative needs an uploaded entry module at ${_sd}" "reviewer_seed_missing"; exit 1; }
            cp "${_sd}/adapter_entry.py" "${SUBMISSION_DIR}/adapter_entry.py"
            head -c 900000 /dev/urandom > "${SUBMISSION_DIR}/adapter.bin"
            echo "[mode] seeded an OVER-BUDGET artifact (expect the byte-budget gate to zero it)" ;;
        negative_stash)
            _sd="${STASH_ORACLE_DIR:-/opt/stash_oracle}"
            [ -f "${_sd}/adapter_entry.py" ] || { emit_zero "stash-oracle negative needs an uploaded entry module at ${_sd}" "reviewer_seed_missing"; exit 1; }
            cp "${_sd}/adapter_entry.py" "${SUBMISSION_DIR}/adapter_entry.py"
            head -c 4096 /dev/urandom > "${SUBMISSION_DIR}/adapter.bin"
            mkdir -p /opt/stash /dev/shm /var/tmp
            head -c 400000 /dev/urandom > /opt/stash/extra.bin
            head -c 400000 /dev/urandom > /dev/shm/stash.bin
            head -c 400000 /dev/urandom > /tmp/stash.bin
            head -c 400000 /dev/urandom > /var/tmp/stash_link.bin
            echo "[mode] seeded the STASH-ORACLE + 4 hidden stashes (expect the sweep to zero it)" ;;
        negative_degenerate)
            _sd="${NEGATIVE_DIR:-/opt/negative}"
            [ -f "${_sd}/adapter_entry.py" ] || { emit_zero "degenerate negative needs an uploaded entry module at ${_sd}" "reviewer_seed_missing"; exit 1; }
            cp "${_sd}/adapter_entry.py" "${SUBMISSION_DIR}/adapter_entry.py"
            head -c 1024 /dev/zero > "${SUBMISSION_DIR}/adapter.bin"
            echo "[mode] seeded a DEGENERATE constant-logit stub (expect the anti-spoof gate to zero it)" ;;
        *) echo "[mode] unknown VERIFIER_MODE=${MODE}, treating as candidate" ;;
    esac
fi

# --- step 2: frozen-surface hard gate. The frozen eval surface is uploaded fresh under /tests at
#     scoring (it is NOT baked into the image any more). Run the FRESH /tests verifier and
#     self-check its sha256 against the manifest. ---
VERIFIER="/tests/compute_reward.py"
[ -f "${VERIFIER}" ] || VERIFIER="/opt/verifier/compute_reward.py"
if [ ! -f "${VERIFIER}" ]; then emit_zero "verifier missing (nothing at /tests/compute_reward.py)" "frozen_surface_missing"; exit 1; fi
MANIFEST="/tests/verifier-correctness-manifest.json"
[ -f "${MANIFEST}" ] || MANIFEST="/opt/verifier/verifier-correctness-manifest.json"
if [ ! -f "${MANIFEST}" ]; then emit_zero "frozen manifest missing (nothing at /tests/verifier-correctness-manifest.json)" "frozen_surface_missing"; exit 1; fi
exp_sha=$("${PY}" -c "import json;print(json.load(open('${MANIFEST}')).get('compute_reward_sha256',''))" 2>/dev/null)
got_sha=$(sha256sum "${VERIFIER}" | awk '{print $1}')
if [ -n "${exp_sha}" ] && [ "${exp_sha}" != "${got_sha}" ]; then
    emit_zero "verifier sha256 mismatch (frozen surface tampered): got ${got_sha}" "forbidden_edit_path"; exit 1
fi

# --- step 3: entry-contract presence. ---
if [ ! -f "${SUBMISSION_DIR}/adapter_entry.py" ]; then emit_zero "missing adapter_entry.py (needs build_adapted_model)" "build_or_entry_contract_failed"; exit 1; fi
if [ ! -f "${SUBMISSION_DIR}/adapter.bin" ]; then emit_zero "missing adapter.bin" "build_or_entry_contract_failed"; exit 1; fi
if [ ! -f "${BASE_MODEL_DIR}/model.safetensors" ]; then emit_zero "frozen base model missing at ${BASE_MODEL_DIR}" "build_import_or_readiness_failed"; exit 1; fi

# --- steps 4-7: the byte-budget gate (dual-measured) + the frozen-base anchor + the
#     anti-spoof quality gate + the held-out CE + the BOUNDED log reward are all computed by
#     the eval-only verifier in ONE isolated pass. `-I` + unset PYTHONPATH/PYTHONHOME +
#     PYTHONSAFEPATH=1 stop the submitted module from shadowing the verifier
#     (sanitize_python_path also runs); the verifier quarantines every solver-writable path
#     for the eval window so the only bytes that travel are the two declared files. ---
cp "${VERIFIER}" /tmp/e2e_a8_verifier.py
unset PYTHONPATH PYTHONHOME
# MKL/OpenMP hardening: a fresh `import torch` after another torch process ran in
# this container can die with "oneMKL FATAL ERROR: Cannot load libtorch_cpu.so".
export MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1
export LD_LIBRARY_PATH="$(dirname "$("${PY}" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib","x"))' 2>/dev/null || echo /x/x)"):${LD_LIBRARY_PATH}"
run_verifier() {
  SUBMISSION_DIR="${SUBMISSION_DIR}" BASE_MODEL_DIR="${BASE_MODEL_DIR}" \
    PYTHONSAFEPATH=1 "${PY}" -I /tmp/e2e_a8_verifier.py
}
run_verifier
verify_exit=$?
if [ "${verify_exit}" -ge 2 ] && [ ! -s /logs/verifier/reward.txt ]; then
  echo "[test] verifier aborted before writing a reward (rc=${verify_exit}); one retry"
  sleep 3; run_verifier; verify_exit=$?
fi
# 🔴 last-resort contract backstop: if the scorer died before writing anything (missing dep /
#    OOM / SIGKILL) the contract would otherwise be EMPTY. Emit all six with a named reason.
if [ ! -s /logs/verifier/reward.txt ]; then
  emit_zero "verifier crashed before writing a reward (rc=${verify_exit})" "verifier_crashed"
  exit 1
fi
echo "[test] compute_reward rc=${verify_exit} ; reward=$(cat /logs/verifier/reward.txt 2>/dev/null)"
exit "${verify_exit}"
