#!/usr/bin/env bash
# Verifier for wre-spec-accept-slot-compact.
# accepted-prefix KV commit: ragged destination plan, global survivor compaction and the verified-step row movement
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"

# ---- anchor (uploaded fresh in tests/, so it can be
# ---- recalibrated without an image rebuild).
# This lane's raw measurement is vs_oracle = oracle_ms / candidate_ms, because the
# start state is an untimeable empty stub. The reward formula instead needs
#   speedup     = baseline_ms / candidate_ms          (vs a correct-but-slow reference)
#   ref_speedup = the oracle's speedup over that same reference
# The correct-but-slow reference here is the baseline2 variant, whose validated mode
# score is b2 = oracle_ms / baseline2_ms, so ref_speedup = 1/b2 (tests/ref_speedup.txt,
# derived from the recorded 4/5-mode measurement) and the baseline timing follows from
# the LIVE in-image oracle_ms:  baseline_ms = oracle_ms * ref_speedup.
# Deriving it live (rather than baking a second constant) keeps the two anchors from
# drifting apart if the image's oracle_ms is ever recalibrated.
ref_speedup=1.0
[ -f "$TESTS_DIR/ref_speedup.txt" ] && ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
base_ms=-1
SCOPE=("accept_compact.py")
MODNAME="accept_compact"
TOKEN="WRE_ACCEPT_RESULT"

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

reward=0.0; vs_oracle=0.0; oracle_ms=-1; cand_ms=-1

# ---- (4) CORRECTNESS GATE: current tree must match the independent reference ----
#      (noop = NotImplementedError -> correctness_ok:false -> gate fails -> reward 0)
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep "$TOKEN" "$LOG/correctness.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys
try: print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: candidate wall + ORACLE anchor (vs_oracle = oracle_ms / cand_ms) ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep "$TOKEN" "$LOG/cand_timing.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys;print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  if [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ]; then
    cp "$REPO/${SCOPE[0]}" "$LOG/cand_scope.bak" 2>/dev/null || true
    git -C "$REPO" checkout -q HEAD -- "${SCOPE[@]}" 2>/dev/null || true
    git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH" 2>"$LOG/anchor_apply.log" || add_hard "anchor_oracle_apply_failed"
    ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/oracle_timing.out" 2>&1 || add_hard "oracle_timing_failed"
    oracle_ms=$(grep "$TOKEN" "$LOG/oracle_timing.out" | TOKEN="$TOKEN" python3 -c "import json,os,sys;print(json.loads(sys.stdin.read().split(os.environ['TOKEN']+' ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
    git -C "$REPO" checkout -q HEAD -- "${SCOPE[@]}" 2>/dev/null || true
    cp "$LOG/cand_scope.bak" "$REPO/${SCOPE[0]}" 2>/dev/null || true
  elif [ -f /opt/verifier-correctness-manifest.json ]; then
    oracle_ms=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('oracle_ms',-1))" 2>/dev/null || echo -1)
  else add_hard "no_oracle_anchor"; fi
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $oracle_ms>0 else 0)")" = "1" ]; then
    vs_oracle=$(python3 -c "print(round($oracle_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict ----
# ---- reward (performance class) ----
# $vs_oracle above is oracle_ms/candidate_ms (the empty-stub start cannot be timed, so
# the raw 1.0 anchor is the ORACLE). Convert it to the reward formula's speedup -- measured
# against the correct-but-slow reference -- and apply the log-ratio formula:
#     baseline_ms = oracle_ms * ref_speedup
#     speedup     = baseline_ms / candidate_ms = $vs_oracle * ref_speedup
#     reward      = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0     in [0.0, 1.0]
# An oracle-grade candidate has $vs_oracle == 1 => speedup == ref_speedup => reward 0.5.
# HARD PRE-GATES (any one => reward 0, the formula is NOT evaluated): every gate
# already collected in HARD (build/import-origin, correctness, scope/anti-cheat,
# timing) PLUS the reward formula's speedup <= 1 and ref_speedup <= 1.
speedup=0.0
if [ "$(python3 -c "print(1 if $vs_oracle>0 and $ref_speedup>0 else 0)" 2>/dev/null || echo 0)" = "1" ]; then
  speedup=$(python3 -c "print(round($vs_oracle * $ref_speedup, 6))" 2>/dev/null || echo 0.0)
  base_ms=$(python3 -c "print(round($oracle_ms * $ref_speedup, 6))" 2>/dev/null || echo -1)
fi
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
export WRO_MODE="$MODE" WRO_REWARD="$reward" WRO_VSORACLE="$vs_oracle" \
       WRO_SPEEDUP="$speedup" WRO_REF="$ref_speedup" WRO_BASE_MS="$base_ms" \
       WRO_ORACLE_MS="$oracle_ms" WRO_CAND_MS="$cand_ms" \
       WRO_HARD="$hard_str" WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" \
       WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
v = {
  "mode": os.environ.get("WRO_MODE"),
  # result-JSON contract (performance class)
  "task_type": "performance",
  "reward": f(os.environ.get("WRO_REWARD")),
  "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; 0 if any hard pre-gate hit",
  "speedup": f(os.environ.get("WRO_SPEEDUP", "0")),   # reward speedup = baseline_ms/cand_ms
  "ref_speedup": f(os.environ.get("WRO_REF", "1")),
  "cv": None,                                         # this harness reports a median, not dispersion
  "vs_oracle": f(os.environ.get("WRO_VSORACLE")),     # raw oracle_ms/cand_ms (dev_speedup reads this)
  "oracle_ms": f(os.environ.get("WRO_ORACLE_MS")),
  "baseline_ms": f(os.environ.get("WRO_BASE_MS", "-1")),
  "candidate_ms": f(os.environ.get("WRO_CAND_MS")),
  "hard_fails": os.environ.get("WRO_HARD", "").split(),
  "hard_fail_reasons": os.environ.get("WRO_HARD", "").split(),   # canonical field name
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WRO_IMP") == "true",
            "correctness_ok": os.environ.get("WRO_CORR") == "true",
            "benchmark_ok": os.environ.get("WRO_BENCH") == "true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
