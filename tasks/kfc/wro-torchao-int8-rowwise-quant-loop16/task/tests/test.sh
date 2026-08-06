#!/usr/bin/env bash
# Verifier for wro-torchao-int8-rowwise-quant (torchao rowwise int8 quantization for quantized training).
# Type-2 Long-horizon, ACCELERATION lane. Scope = 1 file: torchao/prototype/quantized_training/int8.py
# Modes: candidate | noop | oracle | negative.
# reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 over the wall speedup base_ms/cand_ms
# oracle -> 0.5, ref_speedup^2 -> 1.0 cap, noop -> 0.
# (baseline2 = N/A: single algorithmic lever — vectorized reduction/cast; see scope card.)
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("torchao/prototype/quantized_training/int8.py")

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; ref_speedup=1.0; base_ms=-1; cand_ms=-1

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }
RUNDIR="$TESTS_DIR"

# ---- apply oracle / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]   && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];   then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"   2>"$LOG/oracle_apply.log"   || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true; fi

# ---- (1) scope-diff HARD GATE: only the scope file may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$f" = "$s" ] && keep=true; done
  case "$f" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$f"; }
done < "$LOG/changed_files.txt"

# ---- (2) import-origin assert: torchao under test must be the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import os, sys; sys.path.insert(0, "/app/repo")
import torchao.prototype.quantized_training.int8 as m
print("I8_LOC", m.__file__)
sys.exit(0 if m.__file__.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- ref_speedup: the ORACLE's speedup over the frozen degraded baseline.
# This is the reward anchor: reward = min(1, ln(speedup/ref_speedup)/ln(ref_speedup)).
# The in-image manifest wins; tests/ref_speedup.txt is the uploaded-fresh fallback.
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
if [ "$(python3 -c "print(1 if float('${ref_speedup:-1}')<=1.0 else 0)" 2>/dev/null || echo 1)" = "1" ] && [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
fi
ref_speedup=${ref_speedup:-1.0}

# ---- (4) CORRECTNESS GATE: scope output must match the independent fp32 reference ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$RUNDIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep WRO_GDN_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_GDN_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_GDN_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_GDN_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_GDN_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_GDN_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ]; then
    speedup=$(python3 -c "print(round($base_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict ----
# reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, computed in
# python below. speedup here is ALREADY absolute (base_ms/cand_ms vs the frozen degraded
# baseline). Every hard gate is preserved verbatim.
hard_str=""; nhard="${#HARD[@]}"; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRO_MODE="$MODE" WRO_SPEEDUP="$speedup" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_HARD="$hard_str" WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" \
       WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json, math
def f(x):
    try: return float(x)
    except Exception: return -1.0
ref = f(os.environ.get("WRO_REF", "1"))       # oracle speedup over the degraded baseline
sp = f(os.environ.get("WRO_SPEEDUP", "0"))    # candidate absolute speedup
hard = os.environ.get("WRO_HARD", "").split()
gates = {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
         "import_origin_ok": os.environ.get("WRO_IMP") == "true",
         "correctness_ok": os.environ.get("WRO_CORR") == "true",
         "benchmark_ok": os.environ.get("WRO_BENCH") == "true"}
# ---- HARD pre-gates: any one hit => reward 0, formula NOT entered ----
if not all(gates.values()) and not hard:
    hard.append("gate_failed")
if not math.isfinite(ref) or ref <= 1.0:
    if "ref_speedup_invalid" not in hard: hard.append("ref_speedup_invalid")
if not math.isfinite(sp) or sp <= 1.0:
    if "speedup_not_above_baseline" not in hard: hard.append("speedup_not_above_baseline")
reward = 0.0
if not hard and gates["correctness_ok"]:
    try:
        reward = max(0.0, min(1.0, max(0.0, min(1.0, math.log(sp) / math.log(ref) - 1.0))))
    except Exception:
        reward = 0.0
        hard.append("reward_computation_failed")
v = {
  "task_type": "performance",
  "mode": os.environ.get("WRO_MODE"),
  "reward": reward,
  "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
  "speedup": sp,
  "ref_speedup": ref,
  "cv": {},
  "baseline_ms": f(os.environ.get("WRO_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "metadata": {"vs_oracle_ratio": (sp / ref) if ref > 0 else None},
  "hard_fails": hard,
  "hard_fail_reasons": hard,
  "gates": gates}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
