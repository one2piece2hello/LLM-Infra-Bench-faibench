#!/usr/bin/env bash
# Verifier for wro-fla-wallattn-flash-sol — Type-2 Long-horizon, B2 BEAT (acceleration), GPU H20.
# Subsystem = an windowed decay attention forward (fla.ops.wall_attn.parallel_wall_attn).
# Scope = fla/ops/wall_attn/parallel.py. Baked (noop) = a correct-but-SLOW implementation that evaluates
# the full O(T^2) dense pairwise score tensor with no window skipping.
# The solver must restore an implementation whose run time no longer grows linearly with the
# number of dependent time steps to beat the oracle. reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; perf_metric = speedup
# (compute/step bound; roofline in the scope card). Modes: candidate | noop | oracle | negative.
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
# pin the fla sitecustomize layer + baked repo (non-login exec has no torch on bare python3)
export PYTHONPATH="/opt/wro_sitecustomize:/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"; export WRO_LOG="$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
HG="fla/ops/wall_attn"
SCOPE=("$HG/parallel.py")
# banned tokens: a submitted scope file routing to a prebuilt/turnkey recurrence or state-space
# kernel instead of implementing the windowed decay attention itself (W17/D7). DIFF-BASED (only ADDED lines
# count). Comments count. NOTE: triton / triton.language are NOT banned -- writing a custom
# kernel in the scope file is a legitimate reimplementation; the ban targets turnkey ops from
# OTHER modules (incl. this package's own naive_wall_attn reference wrapper).
BANNED=("naive_wall_attn" "torch.ops")

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; import_origin_ok=true; benchmark_ok=true; ban_ok=true
corr_frac=0.0; base_ms=-1; cand_ms=-1; ref_speedup=1.0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; }

# ---- apply oracle / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]    && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];    then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"    2>"$LOG/oracle_apply.log"    || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ]  && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ];  then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH"  2>"$LOG/negative_apply.log"  || true; fi

# ---- (1) scope-diff HARD GATE: only the scope file(s) may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r ff; do
  [ -z "$ff" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$ff" = "$s" ] && keep=true; done
  case "$ff" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$ff"; }
done < "$LOG/changed_files.txt"

# ---- (2) vendor-op ban: candidate's ADDED lines must not route to a turnkey recurrence ----
for s in "${SCOPE[@]}"; do
  [ -f "$REPO/$s" ] || continue
  git -C "$REPO" diff HEAD -- "$s" 2>/dev/null | grep '^+' | grep -v '^+++' > "$LOG/added_lines.txt" || true
  for tok in "${BANNED[@]}"; do
    if grep -Fq "$tok" "$LOG/added_lines.txt"; then ban_ok=false; add_hard "vendor_op_ban:$tok"; fi
  done
done

# ---- (3) import-origin assert: fla under test must be the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import fla, os, sys
loc = os.path.dirname(os.path.abspath(fla.__file__))
print("PKG_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (4) trusted-restore: harness uploaded fresh (never baked) ----
for fn in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$fn" ] || { add_hard "hidden_supplement_missing:$fn"; }
done

# ---- ref_speedup (oracle-calibrated, baked at build; draft pre-bake) ----
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

# ---- clear stale bytecode so the edited/patched scope is the one imported ----
find /app/repo/fla/ops/wall_attn -name "*.pyc" -delete 2>/dev/null || true
find /app/repo/fla/ops/wall_attn -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ---- (5) CORRECTNESS GATE (graded): scope output must match the fp32 reference over the suite ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ] && [ "$ban_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_frac=$(grep WRO_WALLATTN_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_WALLATTN_RESULT ',1)[1]).get('correctness_frac',0.0))
except Exception: print('0.0')" 2>/dev/null || echo 0.0)
fi

# ---- (6) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) — only if fully correct ----
if [ "$(python3 -c "print(1 if $corr_frac>=1.0 else 0)" 2>/dev/null || echo 0)" = "1" ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_WALLATTN_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_WALLATTN_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp "$LOG/cand_timing.out" "$LOG/cand_timing_sol.out" 2>/dev/null || true
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  find /app/repo/fla/ops/wall_attn -name "*.pyc" -delete 2>/dev/null || true
  find /app/repo/fla/ops/wall_attn -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_WALLATTN_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_WALLATTN_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ] || { benchmark_ok=false; add_hard "timing_invalid"; }
fi

# ---- delegate to compute_reward.py (log-ratio reward) ----
export WRO_MODE="$MODE" WRO_HARD="${HARD[*]:-}" WRO_CORR_FRAC="$corr_frac" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" WRO_BENCH="$benchmark_ok" WRO_BAN="$ban_ok"
python3 "$TESTS_DIR/compute_reward.py"
