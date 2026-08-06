#!/usr/bin/env bash
# Verifier for chunked-mlp-recompute. GPU task (anchor calibrated on H20).
# Scope = submission/kernel.py (the ONLY solver-editable file). The solver implements the manual
# forward + input-gradient backward of a gated (SwiGLU) MLP block to the disclosed contract, then
# minimizes the PEAK GPU MEMORY of the call.
#
# task_class = 实现类 (IMPLEMENTATION). Reward is BINARY per
# the reward specification:
#     reward = 1.0  iff EVERY visible correctness case passes AND no cheating
#     reward = 0.0  otherwise (any case fail, or any anti-cheat / forbidden-path hit)
# The former graded ratio (vs_oracle = oracle_peak/candidate_peak) is NO LONGER the reward;
# peak_bytes is still measured and REPORTED (peak_bytes / vs_oracle fields) as diagnostics,
# because the memory objective is what makes the task hard — but it cannot earn partial credit.
# workload.py timing mode reports the peak allocator high-water
# (torch.cuda.max_memory_allocated) in the `timing_ms` field; oracle_ms is a calibrated
# held-out constant (the empty/stub start cannot be measured).
# noop(stub) = NotImplementedError -> correctness fails -> 0.
#
# Modes: candidate | noop | oracle | negative | baseline2.  loop16 calls candidate only.
# Emits ONE single-line JSON verdict: {gates, hard_fails, reward, speedup, ...}  (score_engine §X).
set -uo pipefail
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
git config --global --add safe.directory '*' 2>/dev/null || true   # root-owned /app/repo

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
SUB="submission/kernel.py"
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; oracle_ms=-1; cand_ms=-1; cases_passed=0; cases_total=0; vs_oracle=0.0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }

# ---- oracle_ms: calibrated held-out constant (baked at build; draft default pre-calibration) ----
if [ -f /opt/verifier-correctness-manifest.json ]; then
  oracle_ms=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('oracle_ms',-1))" 2>/dev/null || echo -1)
fi

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

# ---- (4) CORRECTNESS GATE (HARD): scope output must match the fp32 reference. noop/negative FAIL. ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  read cok cases_passed cases_total < <(grep WRE_RESULT "$LOG/correctness.out" | tail -1 | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1])
    print(d.get('correctness_ok'), d.get('cases_passed',0), d.get('cases_total',0))
except Exception: print('False 0 0')" 2>/dev/null || echo "False 0 0")
  cases_passed=${cases_passed:-0}; cases_total=${cases_total:-0}
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
  # reward.md implementation class: EVERY case must pass — a short suite is a FAIL,
  # never partial credit.
  if [ "$cases_total" -gt 0 ] && [ "$cases_passed" -lt "$cases_total" ]; then
    correctness_ok=false
    case " ${HARD[*]:-} " in *" correctness_failed "*) : ;; *) add_hard "correctness_cases_incomplete" ;; esac
  fi
fi

# ---- (5) TIMING (only if correct): candidate_ms = peak_bytes; speedup = oracle_ms / candidate_ms ----
if [ "$correctness_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  read cand_ms flat_ok stable_ok < <(grep WRE_RESULT "$LOG/cand_timing.out" | tail -1 | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read().split('WRE_RESULT ',1)[1]); print(d.get('timing_ms',-1), d.get('flat_ok',False), d.get('stable_ok',False))
except Exception: print('-1 False False')" 2>/dev/null || echo "-1 False False")
  # NOTE: flat_ok/stable_ok are RECORDED diagnostics, not a hard gate — the CSPRNG cache-probe
  # in the correctness stage is the real anti-cache gate (it subsumes constant-output cheats).
  # Only an invalid measurement (candidate_ms<=0) hard-fails timing. (peak_bytes metric: cand_ms is
  # the peak GPU bytes; lower is better; vs_oracle = oracle_bytes/cand_bytes works unchanged.)
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $oracle_ms>0 else 0)")" = "1" ]; then
    vs_oracle=$(python3 -c "print(round($oracle_ms/$cand_ms,6))")
    speedup="$vs_oracle"     # kept for the loop harness's dev_speedup field (§X); NOT the reward
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict (SINGLE-LINE JSON; §X score_engine parses the last {gates,hard_fails} line) ----
# ---- BINARY reward (implementation class) ----
# 1.0 iff every visible case passed and no gate/anti-cheat/forbidden-path fired; else 0.0.
# The measured peak-memory ratio is reported but NEVER scaled into the reward.
reward=0.0
nhard="${#HARD[@]}"
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ] && [ "$benchmark_ok" = true ]; then
  reward=1.0
fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRE_MODE="$MODE" WRE_REWARD="$reward" WRE_SPEEDUP="$speedup" \
       WRE_VSORACLE="$vs_oracle" WRE_CASES_PASSED="$cases_passed" WRE_CASES_TOTAL="$cases_total" \
       WRE_ORACLE_MS="$oracle_ms" WRE_CAND_MS="$cand_ms" \
       WRE_HARD="$hard_str" WRE_SCOPE="$scope_ok" WRE_IMP="$import_origin_ok" \
       WRE_CORR="$correctness_ok" WRE_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except Exception: return -1.0
def i(x):
    try: return int(x)
    except Exception: return 0
sp = f(os.environ.get("WRE_SPEEDUP","0"))
v = {
  "mode": os.environ.get("WRE_MODE"),
  # reward.md result-JSON contract (implementation class)
  "task_type": "implementation",
  "reward": f(os.environ.get("WRE_REWARD")),
  "reward_formula": "binary: 1.0 iff all cases pass and no cheating, else 0.0",
  "tests": {"passed": i(os.environ.get("WRE_CASES_PASSED","0")),
            "total":  i(os.environ.get("WRE_CASES_TOTAL","0"))},
  # peak-memory diagnostics — measured and reported, but NOT scaled into the reward
  "peak_bytes": f(os.environ.get("WRE_CAND_MS")),
  "oracle_peak_bytes": f(os.environ.get("WRE_ORACLE_MS")),
  "speedup": sp,                                   # dev_speedup reads this (§X); diagnostic only
  "vs_oracle": f(os.environ.get("WRE_VSORACLE","0")),
  "oracle_ms": f(os.environ.get("WRE_ORACLE_MS")),
  "candidate_ms": f(os.environ.get("WRE_CAND_MS")),
  "hard_fails": os.environ.get("WRE_HARD","").split(),
  "hard_fail_reasons": os.environ.get("WRE_HARD","").split(),   # reward.md field name
  "gates": {"scope_ok": os.environ.get("WRE_SCOPE")=="true",
            "import_origin_ok": os.environ.get("WRE_IMP")=="true",
            "correctness_ok": os.environ.get("WRE_CORR")=="true",
            "benchmark_ok": os.environ.get("WRE_BENCH")=="true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
