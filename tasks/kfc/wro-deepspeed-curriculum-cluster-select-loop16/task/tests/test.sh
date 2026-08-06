#!/usr/bin/env bash
# Verifier for wro-deepspeed-curriculum-cluster-select (curriculum-learning
# difficulty-cluster selection for large-scale training stability).
# ACCELERATION, CPU (host logic; no GPU).
# Scope = curriculum_cluster.py (select_curriculum_cluster).
# Modes: candidate | noop | oracle | negative | baseline2.
# reward:
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0,  speedup = base_ms/cand_ms (raw wall).
# An oracle-grade candidate scores 0.5; noop (speedup ~ 1.0, degraded vs itself) hits the
# speedup<=1 pre-gate -> 0. The 4-mode speedups quoted below are RAW ratios, pre-formula:
# oracle >> 1 (vectorized boundary-search selection vs O(rows^2) per-row Python
# concatenate); baseline2 in between (block-buffered concat, still row-walked);
# negative = 0 (whole-row shortcut mis-selects the partial boundary rows).
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("curriculum_cluster.py")

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; ref_speedup=1.0; base_ms=-1; cand_ms=-1

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }

# ---- apply oracle / baseline2 / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]    && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];    then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"    2>"$LOG/oracle_apply.log"    || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "baseline2" ] && [ -n "${KERNELBENCH_BASELINE2_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_BASELINE2_PATCH" 2>"$LOG/baseline2_apply.log" || add_hard "baseline2_apply_failed"; fi
if [ "$MODE" = "negative" ]  && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ];  then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH"  2>"$LOG/negative_apply.log"  || true; fi

# ---- (1) scope-diff HARD GATE: only the scope file may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$f" = "$s" ] && keep=true; done
  case "$f" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$f"; }
done < "$LOG/changed_files.txt"

# ---- (2) import-origin assert: the scope module resolves to the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import os, sys
sys.path.insert(0, "/app/repo")
import curriculum_cluster as m
loc = os.path.realpath(m.__file__)
print("SCOPE_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- ref_speedup (oracle-calibrated; metadata only) ----
# ---- ref_speedup (oracle-calibrated anchor for the reward formula) ----
# ref_speedup is LOAD-BEARING (the log-ratio reward formula),
# not metadata. Prefer the tests/-local calibrated value (uploaded fresh at scoring,
# so it can be recalibrated without an image rebuild); fall back to the in-image
# manifest. NOTE: on this lane the baked manifest often carries an UNCALIBRATED 1.0,
# which is why tests/ref_speedup.txt takes precedence.
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi

# ---- (4) CORRECTNESS GATE: scope output must match the independent reference ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep WRO_CURRIC_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_CURRIC_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_CURRIC_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_CURRIC_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp "$REPO/${SCOPE[0]}" /tmp/wro_cand_scope.py 2>/dev/null || true
  git -C "$REPO" checkout -q HEAD -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_CURRIC_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_CURRIC_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp /tmp/wro_cand_scope.py "$REPO/${SCOPE[0]}" 2>/dev/null || true
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ]; then
    speedup=$(python3 -c "print(round($base_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict ----
# reward (performance class):
#   reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0,  strictly in [0.0, 1.0]
#   matching the oracle => 0.5; reaching ref_speedup^2 => 1.0
# HARD PRE-GATES (any one hit => reward 0, formula NOT evaluated): build/import
# failure, ANY correctness case fail, cheating, forbidden edit path, speedup <= 1,
# ref_speedup <= 1. The pre-existing gates below are UNCHANGED; two reward-formula
# pre-gates are added as hard fails (speedup<=1, ref_speedup<=1).
if [ "$correctness_ok" = true ]; then
  if [ "$(python3 -c "print(1 if $speedup>1.0 else 0)" 2>/dev/null || echo 0)" != "1" ]; then
    add_hard "speedup_not_above_baseline"
  fi
  if [ "$(python3 -c "print(1 if $ref_speedup>1.0 else 0)" 2>/dev/null || echo 0)" != "1" ]; then
    add_hard "ref_speedup_invalid"
  fi
fi
reward=0.0
nhard="${#HARD[@]}"
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ]; then
  reward=$(python3 -c "
import math
sp=$speedup; ref=$ref_speedup
print(round(max(0.0, min(1.0, max(0.0, min(1.0, math.log(sp) / math.log(ref) - 1.0)))), 6))" 2>/dev/null || echo 0.0)
fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRO_MODE="$MODE" WRO_REWARD="$reward" WRO_SPEEDUP="$speedup" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_HARD="$hard_str" WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" \
       WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
ref = f(os.environ.get("WRO_REF", "1")); sp = f(os.environ.get("WRO_SPEEDUP", "0"))
v = {
  "mode": os.environ.get("WRO_MODE"),
  # result-JSON contract (performance class)
  "task_type": "performance",
  "reward": f(os.environ.get("WRO_REWARD")),
  "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; 0 if any hard pre-gate hit",
  "speedup": sp,
  "baseline_ms": f(os.environ.get("WRO_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "ref_speedup": ref,
  "metadata": {"vs_oracle_ratio": (sp/ref) if ref > 0 else None},
  "hard_fails": os.environ.get("WRO_HARD", "").split(),
  "hard_fail_reasons": os.environ.get("WRO_HARD", "").split(),   # canonical field name
  "cv": None,                                                    # this harness reports a median, not dispersion
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WRO_IMP") == "true",
            "correctness_ok": os.environ.get("WRO_CORR") == "true",
            "benchmark_ok": os.environ.get("WRO_BENCH") == "true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
