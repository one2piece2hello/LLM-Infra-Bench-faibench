#!/bin/bash
# e2e-a3-moe-train-budget verifier orchestration (FROZEN eval surface).
#
# Single-shot family-A: run the solver's training UNDER a harness-owned wall-clock timer
# (external process-group kill — never a cooperative in-loop timer, never a self-reported
# step/token count), then eval the produced checkpoint.
#
# REWARD (the bench reward spec, performance class, BOUNDED):
#     speedup     = baseline_bpb / candidate_val_bpb        (quality ratio at a FIXED budget)
#     ref_speedup = baseline_bpb / oracle_val_bpb           (FROZEN authoring constant)
#     reward      = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0            in [0, 1]
# Matching the BASELINE recipe scores 0 (pre-gate 5); matching the demonstrated in-budget
# ceiling scores 0.5. The scorer reads the two constants from the frozen manifest and NEVER
# runs the oracle or the baseline.
#
# The whole /app/repo (nanoGPT) + /app/submission is editable; the eval surface (/tests/*) is
# frozen and uploaded FRESH at scoring (nothing is baked into the image).
set -o pipefail

# Lead PATH with the base image's uv venv (/opt/kernelbench-venv has torch + sentencepiece; a
# bare system python3 does NOT). the harness may exec login-style and resets PATH, so pin it.
export PATH=/opt/kernelbench-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Pin BLAS/OMP thread counts so runtime does not depend on node topology (measured: a 145-core
# node let torch open 96 threads and ran 4.3x SLOWER than a pinned single-thread host loop).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

# 🔴 Strip every inherited E2E_* variable FIRST. The container entrypoint is a LOGIN shell that
# sources the solver-owned ~/.bashrc, so anything E2E_* arriving from the environment is
# solver-controlled. Only values this script derives from the FROZEN manifest are honoured.
for v in $(env | sed -n 's/^\(E2E_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset VERIFIER_MODE KERNELBENCH_VERIFY_MODE

FROZEN=/tests
MANIFEST="$FROZEN/verifier-correctness-manifest.json"
SUBMISSION_DIR=/app/submission
VDIR=/logs/verifier
mkdir -p "$VDIR" /logs/artifacts
rm -f "$VDIR"/reward.txt "$VDIR"/reward.json "$VDIR"/metrics.json \
      "$VDIR"/verifier_state.json "$VDIR"/correctness_results.json "$VDIR"/benchmark_results.json
cd /            # a deleted CWD kills torch's dlopen; anchor somewhere that always exists

# The complete verifier output contract on EVERY hard-fail path. Delegated to the frozen
# scorer's --emit-zero so the hard-fail schema can never drift from the success schema; the
# inline fallback covers the case where python itself cannot start.
PY=/opt/kernelbench-venv/bin/python3
[ -x "$PY" ] || PY=python3
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
  echo "{\"task_id\":\"e2e-a3-moe-train-budget\",\"task_type\":\"performance\",\"reward\":0.0,\"hard_fail_reasons\":[\"${_r}\"],\"detail\":\"${_msg}\"}" > "$VDIR/verifier_state.json"
  echo "{\"passed\":0,\"total\":0,\"expected_total\":6,\"all_passed\":false,\"hard_fail_reasons\":[\"${_r}\"],\"checks\":[{\"name\":\"harness pre-flight\",\"passed\":false,\"message\":\"${_msg}\",\"gate\":\"${_r}\"}]}" > "$VDIR/correctness_results.json"
  echo "{\"metric_kind\":\"quality_at_fixed_budget\",\"metric_name\":\"val_bpb\",\"metric_direction\":\"lower_is_better\",\"timing_measured\":false,\"candidate_val_bpb\":null,\"speedup\":null,\"metric_void_on_hard_fail\":true,\"oracle_executed_by_scorer\":false,\"baseline_executed_by_scorer\":false,\"hard_fail_reasons\":[\"${_r}\"]}" > "$VDIR/benchmark_results.json"
  echo 0.000000 > "$VDIR/reward.txt"
}

save_artifacts() {
  for p in "$SUBMISSION_DIR/train_gpt.py" "$SUBMISSION_DIR/run_training.sh" "$VDIR/train.log"; do
    [ -e "$p" ] && cp -r "$p" "/logs/artifacts/$(basename "$p")" 2>/dev/null || true
  done
}
trap save_artifacts EXIT

# --- the frozen surface must be the FRESH /tests upload, and ONLY that -------------------
# There is deliberately no baked /opt/verifier fallback any more: a baked copy leaks the
# held-out corpus and the calibrated constants into the solver's own container, and lets a
# stale scorer decide a run.
FROZEN_SET="compute_reward.py test.sh timer.sh holdout_corpus.txt verifier-correctness-manifest.json"
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
if [ -e /opt/verifier ]; then
  emit_zero "an in-image /opt/verifier evaluation surface is present (image is stale / re-leaked)" "forbidden_edit_path"; exit 1
fi

# verifier self-integrity against the manifest pin
EXP_SHA=$("$PY" -c "import json;print(json.load(open('$MANIFEST')).get('compute_reward_sha256',''))" 2>/dev/null)
GOT_SHA=$(sha256sum "$FROZEN/compute_reward.py" | awk '{print $1}')
if [ -n "$EXP_SHA" ] && [ "$EXP_SHA" != "$GOT_SHA" ]; then
  emit_zero "verifier sha256 mismatch (frozen surface tampered): expected $EXP_SHA got $GOT_SHA" "forbidden_edit_path"; exit 1
fi

# --- read the frozen budget from the manifest -------------------------------------------
read_man() { "$PY" -c "import json;d=json.load(open('$MANIFEST'));print(d.get('budget',{}).get('$1', d.get('$1','$2')))"; }
WALLCLOCK_SEC=$(read_man wallclock_sec 600)
SEED=$("$PY" -c "import json;print(json.load(open('$MANIFEST')).get('harness_seed',1337))")
PARAM_FLOOR=$("$PY" -c "import json;d=json.load(open('$MANIFEST'));print((d.get('quality_gate') or {}).get('param_floor', d.get('param_floor',150000000)))")
OUT_CKPT="$SUBMISSION_DIR/model_ckpt.pt"
echo "[test] budget: wallclock=${WALLCLOCK_SEC}s param_floor=${PARAM_FLOOR} seed=${SEED} (harness-fixed)"

# --- entry contract ---------------------------------------------------------------------
[ -f "$SUBMISSION_DIR/run_training.sh" ] || { emit_zero "missing entry contract $SUBMISSION_DIR/run_training.sh" "build_or_entry_contract_failed"; exit 1; }
[ -f "$SUBMISSION_DIR/train_gpt.py" ]    || { emit_zero "missing $SUBMISSION_DIR/train_gpt.py (needs load_model_for_verification)" "build_or_entry_contract_failed"; exit 1; }
rm -f "$OUT_CKPT"

# --- fingerprint the frozen surface, then QUARANTINE it for the training phase -----------
# The training entry runs as root in this container, so /tests cannot be made cryptographically
# unreachable; what the harness CAN do is (a) make the held-out corpus and the calibrated
# constants unreadable for the duration of the run and (b) detect any change to them. Both a
# content change and a mode change are red lines -> reward 0 with a named reason.
STATE_DIR=$(mktemp -d /tmp/.e2e_a3_state.XXXXXXXX)
chmod 0700 "$STATE_DIR"
"$PY" - "$FROZEN" "$STATE_DIR/before.json" <<'PYFP'
import hashlib, json, os, sys
root, out = sys.argv[1], sys.argv[2]
rec = {}
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name)
    if not os.path.isfile(p):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(p)
    rec[name] = {"sha256": h.hexdigest(), "mode": st.st_mode & 0o7777, "size": st.st_size}
json.dump({"dir_mode": os.stat(root).st_mode & 0o7777, "files": rec}, open(out, "w"))
PYFP
[ -s "$STATE_DIR/before.json" ] || { emit_zero "could not fingerprint the frozen surface" "build_or_entry_contract_failed"; exit 1; }
# Copy the harness-owned timer OUT of /tests before quarantining, so launching the training
# does not depend on /tests being readable.
cp "$FROZEN/timer.sh" "$STATE_DIR/timer.sh" && chmod 0700 "$STATE_DIR/timer.sh"
SAVED_DIR_MODE=$(stat -c %a "$FROZEN")
chmod 0000 "$FROZEN" 2>/dev/null || true

# --- run the solver's training under the hard wall-clock cap ----------------------------
# DATA_PATH / TOKENIZER_PATH are PINNED by the harness (not inherited from the environment):
# the scored corpus must not be selectable by anything the solver can set (a login shell
# sources ~/.bashrc, so an inherited value is solver-controllable).
export SEED PARAM_FLOOR WALLCLOCK_SEC OUT_CKPT SUBMISSION_DIR
export DATA_PATH=/data/datasets/wikitext_bpe8k
export TOKENIZER_PATH=/data/tokenizers/wikitext_bpe8k.model
export NANOGPT_REPO=/app/repo
export PYTHONPATH=/app/repo
# --- PRIVILEGE DROP for the training phase ------------------------------------------------
# chmod 0000 on /tests means NOTHING to uid 0. Running the candidate's training under an
# unprivileged uid is what makes the held-out quarantine real instead of nominal. Everything the
# training legitimately needs is made accessible to that uid first; the frozen surface is not.
TRAIN_UID=$("$PY" -c "import json;print((json.load(open('$MANIFEST')).get('anti_tamper') or {}).get('training_uid') or '')" 2>/dev/null)
if [ -n "$TRAIN_UID" ]; then
  export E2E_TRAIN_UID="$TRAIN_UID"
  chmod 0777 "$SUBMISSION_DIR" 2>/dev/null || true
  find "$SUBMISSION_DIR" -type f -exec chmod a+r {} + 2>/dev/null || true
  find "$SUBMISSION_DIR" -type d -exec chmod a+rx {} + 2>/dev/null || true
  chmod -R a+rX /app/repo 2>/dev/null || true
  chmod a+rX /data /data/datasets /data/datasets/wikitext_bpe8k /data/tokenizers 2>/dev/null || true
  find /data/datasets/wikitext_bpe8k /data/tokenizers -type f -exec chmod a+r {} + 2>/dev/null || true
  HOMEDIR="$STATE_DIR/trainhome"; mkdir -p "$HOMEDIR"; chmod 0777 "$HOMEDIR" "$STATE_DIR"
  export HOME="$HOMEDIR" TMPDIR="$HOMEDIR" XDG_CACHE_HOME="$HOMEDIR/.cache" \
         TRITON_CACHE_DIR="$HOMEDIR/.triton" TORCHINDUCTOR_CACHE_DIR="$HOMEDIR/.inductor"
  rm -f /app/.timer/train_uid
  echo "[test] training will run as uid $TRAIN_UID (held-out quarantine is enforced by the uid, not just by chmod)"
fi

echo "[test] launching training (SEED=$SEED) ..."
# 🔴 Write the training log to a FILE, not through `| tee`: a surviving detached grandchild that
# holds the pipe open would block this shell forever (MEASURED — the run hung instead of scoring).
bash "$STATE_DIR/timer.sh" "$WALLCLOCK_SEC" bash "$SUBMISSION_DIR/run_training.sh" > "$VDIR/train.log" 2>&1
TRAIN_RC=$?
tail -40 "$VDIR/train.log" 2>/dev/null || true
echo "[test] training returned rc=$TRAIN_RC"
ACTUAL_TRAIN_UID=$(cat /app/.timer/train_uid 2>/dev/null | awk '{print $2}')
echo "[test] training ran as euid=${ACTUAL_TRAIN_UID:-unknown} (requested ${TRAIN_UID:-root})"
# the checkpoint is written by the unprivileged uid; make it readable for the eval phase
chmod a+r "$OUT_CKPT" 2>/dev/null || true

# --- un-quarantine and re-fingerprint ---------------------------------------------------
chmod "$SAVED_DIR_MODE" "$FROZEN" 2>/dev/null || true
"$PY" - "$FROZEN" "$STATE_DIR/before.json" "$STATE_DIR/frozen_state.json" "$SAVED_DIR_MODE" <<'PYCMP'
import hashlib, json, os, sys
root, before_p, out_p, saved_mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
before = json.load(open(before_p))
changed, mode_changed, unexpected = [], [], []
for name, rec in before["files"].items():
    p = os.path.join(root, name)
    if not os.path.isfile(p):
        changed.append(name); continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != rec["sha256"]:
        changed.append(name)
    if (os.stat(p).st_mode & 0o7777) != rec["mode"]:
        mode_changed.append(name)
for name in sorted(os.listdir(root)):
    if os.path.isfile(os.path.join(root, name)) and name not in before["files"]:
        unexpected.append(name)
overrides = sorted(k for k in os.environ if k.startswith("E2E_") and k.endswith("_OVERRIDE"))
json.dump({"n_files": len(before["files"]), "changed": changed, "mode_changed": mode_changed,
           "unexpected_files": unexpected, "overrides_seen": overrides,
           "dir_mode_restored": saved_mode}, open(out_p, "w"))
PYCMP

# --- checkpoint present within the budget? ---------------------------------------------
if [ ! -f "$OUT_CKPT" ]; then
  emit_zero "training produced no checkpoint at $OUT_CKPT within the ${WALLCLOCK_SEC}s budget (rc=$TRAIN_RC)" "build_or_entry_contract_failed"
  exit 1
fi
if [ "$TRAIN_RC" -ne 0 ] && [ "$TRAIN_RC" -ne 124 ]; then
  echo "[test] WARN training rc=$TRAIN_RC (non-timeout); a checkpoint exists, proceeding to eval"
fi

# --- EVAL + bounded reward (isolated process; cannot be shadowed by the submission) -----
# Every E2E_*_OVERRIDE is stripped: on the scored path no environment variable may influence
# the budget, the seed count or the frozen reward constants (a login shell sources ~/.bashrc,
# which the solver owns). Overrides remain available to the AUTHOR's calibration driver,
# which invokes compute_reward.py directly rather than through this script.
for v in $(env | sed -n 's/^\(E2E_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
cp "$FROZEN/compute_reward.py" "$STATE_DIR/compute_reward.py"
unset PYTHONPATH PYTHONHOME
SUBMISSION_DIR="$SUBMISSION_DIR" OUT_CKPT="$OUT_CKPT" \
  E2E_FROZEN_STATE="$STATE_DIR/frozen_state.json" \
  E2E_OBSERVED_TRAIN_UID="${ACTUAL_TRAIN_UID:-}" E2E_REQUESTED_TRAIN_UID="${TRAIN_UID:-}" \
  PYTHONSAFEPATH=1 "$PY" -I "$STATE_DIR/compute_reward.py"
EVAL_RC=$?

# --- output-contract guarantee ----------------------------------------------------------
# The scorer writes all six itself on every path it reaches (its own crash handler included),
# but if the interpreter dies first (missing dep / OOM / SIGKILL) nothing would be written.
CONTRACT_MISSING=false
for f in reward.txt reward.json metrics.json verifier_state.json correctness_results.json benchmark_results.json; do
  [ -s "$VDIR/$f" ] || CONTRACT_MISSING=true
done
if [ "$CONTRACT_MISSING" = true ]; then
  emit_zero "verifier process died before writing the result contract (exit $EVAL_RC)" "verifier_crashed"
  rm -rf "$STATE_DIR"
  exit 1
fi
echo "[test] compute_reward rc=$EVAL_RC ; reward=$(cat "$VDIR/reward.txt" 2>/dev/null)"
rm -rf "$STATE_DIR"
exit "$EVAL_RC"
