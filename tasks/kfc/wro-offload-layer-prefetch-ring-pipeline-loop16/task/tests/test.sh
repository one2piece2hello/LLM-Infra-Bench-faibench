#!/usr/bin/env bash
# Verifier for wro-offload-layer-prefetch-ring-pipeline.
# layer-wise weight prefetch over a pinned staging ring: chunking, lookahead window, ring conflicts, arrivals, stall profile
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("prefetch_ring.py")
MODNAME="prefetch_ring"
TOKEN="WRO_PREFETCHRING_RESULT"

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true

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
MODNAME="$MODNAME" python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import os, sys, importlib
sys.path.insert(0, "/app/repo")
m = importlib.import_module(os.environ["MODNAME"])
loc = os.path.realpath(m.__file__)
print("SCOPE_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

speedup=0.0; ref_speedup=1.0; base_ms=-1; cand_ms=-1
# ---- ref_speedup (oracle-calibrated anchor; the log reward curve needs it) ----
# Prefer the tests/-local override: tests/ is uploaded fresh at scoring time, so the
# anchor can be recalibrated with NO image rebuild. Fall back to the in-image manifest.
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
  [ -n "$ref_speedup" ] || ref_speedup=1.0
elif [ -f "$TESTS_DIR/verifier-correctness-manifest.json" ]; then
  ref_speedup=$(TD="$TESTS_DIR" python3 -c "import json,os;print(json.load(open(os.environ['TD']+'/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi

# ---- (4) CORRECTNESS GATE: scope output must match the independent reference ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep "$TOKEN" "$LOG/correctness.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys
try: print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep "$TOKEN" "$LOG/cand_timing.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys;print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp "$REPO/${SCOPE[0]}" /tmp/wro_cand_scope.py 2>/dev/null || true
  git -C "$REPO" checkout -q HEAD -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep "$TOKEN" "$LOG/base_timing.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys;print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp /tmp/wro_cand_scope.py "$REPO/${SCOPE[0]}" 2>/dev/null || true
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ]; then
    speedup=$(python3 -c "print(round($base_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict ----
# ---- reward: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 ----
# The HARD PRE-GATES are unchanged and still decide 0 first: any hard_fail reason,
# a false correctness gate, speedup <= 1 or ref_speedup <= 1 => reward 0 and the
# formula is NEVER entered. Only a clean, above-baseline run reaches the log curve
# (parity with the oracle scores 0.5; oracle^2 or better caps at 1.0).
reward=0.0
if [ "${#HARD[@]}" -eq 0 ] && [ "$correctness_ok" = true ]; then
  REW_OUT=$(REW_SP="$speedup" REW_REF="$ref_speedup" python3 - <<'PYR'
import math, os
def f(x):
    try: return float(x)
    except Exception: return float("nan")
sp, ref = f(os.environ.get("REW_SP")), f(os.environ.get("REW_REF"))
if not (math.isfinite(sp) and math.isfinite(ref)):
    print("0.0 invalid_primary_metric_value")
elif sp <= 1.0:
    print("0.0 speedup_not_above_baseline")
elif ref <= 1.0:
    print("0.0 ref_speedup_invalid")
else:
    print("%.6f -" % max(0.0, min(1.0, max(0.0, min(1.0, math.log(sp) / math.log(ref) - 1.0)))))
PYR
)
  reward=$(printf '%s\n' "$REW_OUT" | awk 'NR==1{print $1}')
  RGATE=$(printf '%s\n' "$REW_OUT" | awk 'NR==1{print $2}')
  [ -n "${reward:-}" ] || reward=0.0
  if [ -n "${RGATE:-}" ] && [ "$RGATE" != "-" ]; then add_hard "$RGATE"; fi
fi
nhard="${#HARD[@]}"
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
  "reward": f(os.environ.get("WRO_REWARD")),
  "speedup": sp,
  "baseline_ms": f(os.environ.get("WRO_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "ref_speedup": ref,
  "task_type": "performance",
  "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
  "cv": {"baseline": None, "candidate": None},
  "metadata": {"vs_oracle_ratio": (sp/ref) if ref > 0 else None},
  "hard_fails": os.environ.get("WRO_HARD", "").split(),
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WRO_IMP") == "true",
            "correctness_ok": os.environ.get("WRO_CORR") == "true",
            "benchmark_ok": os.environ.get("WRO_BENCH") == "true"}}
v["hard_fail_reasons"] = v["hard_fails"]          # canonical field name (alias)
print(json.dumps(v))
PY
cat "$LOG/reward.json"
