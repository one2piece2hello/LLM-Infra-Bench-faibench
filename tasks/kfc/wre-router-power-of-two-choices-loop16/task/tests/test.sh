#!/usr/bin/env bash
# Verifier for wre-router-power-of-two-choices-loop16.
# Scope = submission/kernel.py (the ONLY solver-editable file). The solver implements the
# "power of two choices" request router (sequential least-loaded-of-two pick) to the disclosed
# contract, then optimizes the per-request host loop.
#
# reward = vs_oracle = oracle_ms / candidate_ms  (raw, uncapped; oracle=1.0, naive<1, faster>1),
# behind a HARD correctness gate (EXACT choices + final_load vs an independent reference). oracle_ms
# is a calibrated held-out constant (the empty/stub start cannot be timed). noop(stub)=NotImplementedError -> 0.
#
# Modes: candidate | noop | oracle | negative | baseline2.  loop16 calls candidate only.
# Emits ONE single-line JSON verdict: {gates, hard_fails, reward, speedup, ...}  (score_engine §X).
set -uo pipefail
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
git config --global --add safe.directory '*' 2>/dev/null || true   # §N crane/root-owned /app/repo

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
SUB="submission/kernel.py"
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; oracle_ms=-1; cand_ms=-1; base_ms=-1; vs_oracle=0.0; ref_speedup=1.0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }

# ---- oracle_ms: calibrated held-out constant (baked at build; draft default pre-calibration) ----
if [ -f /opt/verifier-correctness-manifest.json ]; then
  oracle_ms=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('oracle_ms',-1))" 2>/dev/null || echo -1)
fi
# ---- ref_speedup: the oracle's speedup over the naive-but-correct
# baseline. B1-empty ships no timeable baseline, so this anchor is DERIVED from the two
# already-measured shipped modes (ref = 1/baseline2_speedup) and carried in
# tests/ref_speedup.txt; the baked manifest is the fallback. No new measurement.
ref_speedup=1.0
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi
[ -n "${ref_speedup:-}" ] || ref_speedup=1.0
# baseline_ms implied by the anchor: baseline_ms = oracle_ms * ref_speedup
base_ms=$(python3 -c "print(round($oracle_ms*$ref_speedup,6) if $oracle_ms>0 else -1)" 2>/dev/null || echo -1)

# ---- mode dispatch: swap the reviewer variant over the scope file (NEVER in candidate/noop) ----
case "$MODE" in
  oracle)    [ -n "${KERNELBENCH_ORACLE_FILE:-}" ]    && cp "$KERNELBENCH_ORACLE_FILE"    "$REPO/$SUB" ;;
  negative)  [ -n "${KERNELBENCH_NEGATIVE_FILE:-}" ]  && cp "$KERNELBENCH_NEGATIVE_FILE"  "$REPO/$SUB" ;;
  baseline2) [ -n "${KERNELBENCH_BASELINE2_FILE:-}" ] && cp "$KERNELBENCH_BASELINE2_FILE" "$REPO/$SUB" ;;
  candidate|noop) : ;;   # score the live tree (candidate=solver edit, noop=baked stub)
esac

# ---- (1) scope-diff HARD GATE: only submission/kernel.py may change vs the baked baseline ----
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
      "$SUB") : ;;
      *__pycache__*|*.pyc) : ;;
      *) scope_ok=false; add_hard "out_of_scope_edit:$f" ;;
    esac
  done < "$LOG/changed_files.txt"
fi

# ---- (2) import-origin assert: the scored kernel must be the baked /app/repo copy ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import importlib.util, os, sys
p = "/app/repo/submission/kernel.py"
print("KERNEL_PATH", os.path.realpath(p))
sys.exit(0 if os.path.realpath(p).startswith("/app/repo") and os.path.exists(p) else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked into the solver-visible tree) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- (4) CORRECTNESS GATE (HARD): scope output must match the independent reference. noop/negative FAIL. ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep WRE_RESULT "$LOG/correctness.out" | tail -1 | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING (only if correct): candidate_ms; speedup = oracle_ms / candidate_ms ----
if [ "$correctness_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  read cand_ms flat_ok stable_ok < <(grep WRE_RESULT "$LOG/cand_timing.out" | tail -1 | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1]); print(d.get('timing_ms',-1), d.get('flat_ok',False), d.get('stable_ok',False))
except Exception: print('-1 False False')" 2>/dev/null || echo "-1 False False")
  # NOTE: flat_ok/stable_ok are RECORDED diagnostics, not a hard gate — the CSPRNG cache-probe
  # in the correctness stage is the real anti-cache gate (it subsumes constant-output cheats).
  # Only an invalid measurement (candidate_ms<=0) hard-fails timing.
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $oracle_ms>0 else 0)")" = "1" ]; then
    vs_oracle=$(python3 -c "print(round($oracle_ms/$cand_ms,6))")
    # speedup is measured against the NAIVE BASELINE implied by the anchor, as
    # the reward formula requires (candidate vs baseline), not against the oracle.
    base_ms=$(python3 -c "print(round($oracle_ms*$ref_speedup,6))")
    speedup=$(python3 -c "print(round($base_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict (SINGLE-LINE JSON, §X) ----
# reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 in [0.0, 1.0].
# Pre-gates (any hit -> 0, formula NOT entered): any hard_fail (scope /
# import-origin / trusted-restore / correctness / timing), correctness FAIL,
# speedup <= 1 (did not beat the naive baseline), ref_speedup <= 1.
# No existing gate is relaxed — only the reward arithmetic changed.
reward=0.0
nhard="${#HARD[@]}"
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ] && [ "$benchmark_ok" = true ]; then
  if [ "$(python3 -c "print(1 if $speedup>1.0 else 0)" 2>/dev/null || echo 0)" != "1" ]; then
    add_hard "speedup_not_above_baseline"
  elif [ "$(python3 -c "print(1 if $ref_speedup>1.0 else 0)" 2>/dev/null || echo 0)" != "1" ]; then
    add_hard "ref_speedup_invalid"
  else
    reward=$(python3 -c "import math;print(round(min(1.0,max(0.0,max(0.0, min(1.0, math.log($speedup) / math.log($ref_speedup) - 1.0)))),6))")
  fi
fi
nhard="${#HARD[@]}"
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRE_MODE="$MODE" WRE_REWARD="$reward" WRE_SPEEDUP="$speedup" \
       WRE_ORACLE_MS="$oracle_ms" WRE_CAND_MS="$cand_ms" \
       WRE_REF="$ref_speedup" WRE_BASE_MS="${base_ms:--1}" WRE_VSORACLE="${vs_oracle:-0}" \
       WRE_HARD="$hard_str" WRE_SCOPE="$scope_ok" WRE_IMP="$import_origin_ok" \
       WRE_CORR="$correctness_ok" WRE_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except Exception: return -1.0
sp = f(os.environ.get("WRE_SPEEDUP","0"))
ref = f(os.environ.get("WRE_REF","1"))
hard = os.environ.get("WRE_HARD","").split()
v = {
  "schema_version": "kernelbench_reward_v3_oracle_relative",
  "task_type": "performance",
  "mode": os.environ.get("WRE_MODE"),
  "reward": f(os.environ.get("WRE_REWARD")),
  "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
  "speedup": sp,                                   # candidate vs the naive baseline implied by the anchor
  "ref_speedup": ref,                              # oracle's speedup over that same baseline
  "cv": {"baseline": None, "candidate": None},
  "vs_oracle": f(os.environ.get("WRE_VSORACLE","0")),
  "oracle_ms": f(os.environ.get("WRE_ORACLE_MS")),
  "baseline_ms": f(os.environ.get("WRE_BASE_MS")),
  "candidate_ms": f(os.environ.get("WRE_CAND_MS")),
  "hard_fails": hard,
  "hard_fail_reasons": hard,
  "gates": {"scope_ok": os.environ.get("WRE_SCOPE")=="true",
            "import_origin_ok": os.environ.get("WRE_IMP")=="true",
            "correctness_ok": os.environ.get("WRE_CORR")=="true",
            "benchmark_ok": os.environ.get("WRE_BENCH")=="true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
