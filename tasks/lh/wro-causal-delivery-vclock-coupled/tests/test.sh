#!/usr/bin/env bash
# Verifier for wro-causal-delivery-vclock-coupled -- Type-2 Long-horizon, B2 BEAT.
# Subsystem = causal-broadcast delivery layer under causal/ (pending store + delivery driver, sharing a fixed vector clock).
#   sharing a fixed log-bucket scheme). Scope (editable) = causal/buffer.py +
#   causal/channel.py. vclock.py + __init__.py are out of scope (read-only contract).
# Baseline (baked, noop) = the repo's OWN correct-but-slow path: RouteLatency re-scans its raw
#   sample list to build a histogram on every query (O(N) per query) and SloTracker.global_quantile
#   re-scans every route on every call (O(sum_N) per call). The solver restores an incremental
#   per-bucket histogram (O(1) record, O(buckets) query) coupling both files, to beat the oracle
#   (log-ratio reward, see compute_reward.py), preserving the documented contract. Modes: candidate|noop|oracle|negative|baseline2.
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
git config --global --add safe.directory '*' 2>/dev/null || true
export PYTHONPATH=/app/repo:${PYTHONPATH:-}
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export TESTS_DIR
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"; export WRO_LOG="$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("causal/buffer.py" "causal/channel.py")

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; import_origin_ok=true; benchmark_ok=true
corr_frac=0.0; base_ms=-1; cand_ms=-1; ref_speedup=1.0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; }

# ---- apply oracle / negative / baseline2 patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]    && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];    then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"    2>"$LOG/oracle_apply.log"    || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ]  && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ];  then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH"  2>"$LOG/negative_apply.log"  || true; fi
if [ "$MODE" = "baseline2" ] && [ -n "${KERNELBENCH_BASELINE2_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_BASELINE2_PATCH" 2>"$LOG/baseline2_apply.log" || add_hard "baseline2_apply_failed"; fi

# ---- (1) scope-diff HARD GATE: only the scope file(s) may change in /app/repo ----
git -C "$REPO" update-index -q --refresh 2>/dev/null || true
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r ff; do
  [ -z "$ff" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$ff" = "$s" ] && keep=true; done
  case "$ff" in
    *__pycache__*|*.pyc) keep=true;;
    gated_gelu.py) keep=true;;   # stray artifact inherited from the shared build base; gitignored, not part of the task tree
  esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$ff"; }
done < "$LOG/changed_files.txt"

# ---- (2) import-origin assert: the causal package under test is the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import os, sys
sys.path.insert(0, os.environ["TESTS_DIR"])
import workload
loc = os.path.realpath(workload.scope_pkg().channel.__file__)
print("CHANNEL_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for fn in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$fn" ] || { add_hard "hidden_supplement_missing:$fn"; }
done

# ---- ref_speedup (oracle-calibrated, baked at build; draft pre-bake) ----
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

# ---- (4) CORRECTNESS GATE: pass-rate over MANY diverse cases vs an independent reference ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  find /app/repo -name "*.pyc" -delete 2>/dev/null || true
  find /app/repo -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_frac=$(grep WRO_CAUSAL_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_CAUSAL_RESULT ',1)[1]).get('correctness_frac',0.0))
except Exception: print('0.0')" 2>/dev/null || echo 0.0)
fi

# ---- (5) TIMING: candidate wall vs baseline (frozen baked tree at HEAD) -- only if fully correct ----
if [ "$(python3 -c "print(1 if $corr_frac>=1.0 else 0)" 2>/dev/null || echo 0)" = "1" ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_CAUSAL_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_CAUSAL_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_CAUSAL_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_CAUSAL_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ] || { benchmark_ok=false; add_hard "timing_invalid"; }
fi

# ---- delegate to compute_reward.py (log-ratio reward) ----
export WRO_MODE="$MODE" WRO_HARD="${HARD[*]:-}" WRO_CORR_FRAC="$corr_frac" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" WRO_BENCH="$benchmark_ok"
python3 "$TESTS_DIR/compute_reward.py"
