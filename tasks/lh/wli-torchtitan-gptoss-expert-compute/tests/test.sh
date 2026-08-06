#!/usr/bin/env bash
# Verifier for wli-torchtitan-gptoss-expert-compute -- a torchtitan gpt-oss MoE EXPERT-COMPUTE
# correctness task with a GRADED reward, COUPLED multi-file scope, DEGRADED start-shape (the
# shipped tree RUNS but is functionally degraded at 6 coupled points).
# Scope = torchtitan/models/common/moe.py + torchtitan/models/gpt_oss/moe.py.
# Modes: candidate | noop | oracle | negative | baseline2.
#   reward = weighted hidden-case fraction in [0,1] iff (scope_ok && import_origin_ok),
#   else 0.0. perf_metric = none. Single-process, CPU-only, no GPU, no model weights.
set -uo pipefail
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/opt/kernelbench-venv/bin:$PATH
git config --global --add safe.directory '*' 2>/dev/null || true
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; import_origin_ok=true; harness_ok=true
cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; harness_ok=false; }

# ---- reviewer modes apply a reviewer-only patch onto the shipped (degraded) tree ----
if [ "$MODE" = "oracle" ]    && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"    2>"$LOG/apply.log" || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "baseline2" ] && [ -n "${KERNELBENCH_BASELINE2_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_BASELINE2_PATCH" 2>"$LOG/apply.log" || add_hard "baseline2_apply_failed"; fi
if [ "$MODE" = "negative" ]  && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH"  2>"$LOG/apply.log" || true; fi

# ---- (1) scope-diff HARD GATE: only the 2 declared scope files may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    torchtitan/models/common/moe.py) : ;;
    torchtitan/models/gpt_oss/moe.py) : ;;
    *__pycache__*|*.pyc) : ;;
    *) scope_ok=false; add_hard "out_of_scope_edit:$f" ;;
  esac
done < "$LOG/changed_files.txt"

# ---- (2) harness completeness ----
[ -f "$TESTS_DIR/workload.py" ] || { add_hard "hidden_supplement_missing:workload.py"; harness_ok=false; }

# ---- (3) GRADED GATE (references computed in-harness; import-origin asserted here) ----
score=0.0
if [ "$harness_ok" = true ] && [ "$scope_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 workload.py ) > "$LOG/correctness.out" 2>&1
  read_score=$(python3 - <<'PY'
import json
score = 0.0; origin_ok = False; passed = -1; total = -1
try:
    lines = open("/logs/verifier/correctness.out").read().splitlines()
    d = None
    for line in lines:
        if line.startswith("WRO_RESULT "):
            d = json.loads(line[len("WRO_RESULT "):])
            break
    if d is not None:
        score = float(d.get("score", 0.0) or 0.0)
        origin_ok = bool(d.get("origin_ok", False))
        cs = d.get("cases")
        if isinstance(cs, list):
            total = len(cs)
            passed = sum(1 for c in cs if isinstance(c, dict) and c.get("ok"))
        elif isinstance(cs, dict):
            total = len(cs)
            passed = sum(1 for v in cs.values() if v == "PASS" or (isinstance(v, str) and v.startswith("pass")))
    if total < 0:
        # per-case detail not on the WRO_RESULT line -> some tasks' _emit() helper prints a
        # SECOND bare-JSON line (not WRO_RESULT-prefixed) carrying "cases"; scan for it.
        for line in lines:
            s = line.strip()
            if not s or s.startswith("WRO_RESULT") or s.startswith("ORIGIN_DBG"):
                continue
            try:
                d2 = json.loads(s)
            except Exception:
                continue
            if not isinstance(d2, dict):
                continue
            cs = d2.get("cases")
            if isinstance(cs, dict):
                total = len(cs)
                passed = sum(1 for v in cs.values() if v == "PASS" or (isinstance(v, str) and v.startswith("pass")))
                break
            if isinstance(cs, list):
                total = len(cs)
                passed = sum(1 for c in cs if isinstance(c, dict) and c.get("ok"))
                break
except Exception:
    pass
print(f"{score}|{'1' if origin_ok else '0'}|{passed}|{total}")
PY
)
  IFS='|' read -r score oflag passed total <<< "$read_score"
  [ "$oflag" = "1" ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }
else
  score=0.0; passed=-1; total=-1
fi

# ---- verdict ----
nhard="${#HARD[@]}"
# implementation-class: BINARY reward. 1.0 iff every visible case passed (passed==total,
# total>0) AND no hard-fails/scope/import issues; a single failing case (or unknown case
# data) => 0.0. Falls back to the weighted score only if passed/total could not be derived.
if [ "${total:-0}" -gt 0 ] 2>/dev/null && [ "$passed" = "$total" ]; then
  reward="1.0"
elif [ "${total:-0}" -gt 0 ] 2>/dev/null; then
  reward="0.0"
else
  reward=$(python3 -c "print('1.0' if float('${score:-0}' or 0) >= 1.0 - 1e-6 else '0.0')" 2>/dev/null || echo 0.0)
fi
if [ "$nhard" -gt 0 ] || [ "$import_origin_ok" != true ] || [ "$scope_ok" != true ]; then reward="0.0"; fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRO_MODE="$MODE" WRO_REWARD="$reward" WRO_SCORE="$score" WRO_HARD="$hard_str" \
       WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" WRO_PASSED="$passed" WRO_TOTAL="$total"
python3 - <<'PY'
import os, json
def f(x):
    try: return float(x)
    except Exception: return -1.0
def i(x):
    try: return max(0, int(float(x)))
    except Exception: return 0
hard_list = os.environ.get("WRO_HARD","").split()
verdict = {
  "mode": os.environ.get("WRO_MODE"),
  "task_type": "implementation",
  "reward": f(os.environ.get("WRO_REWARD")),
  "raw_score": f(os.environ.get("WRO_SCORE")),
  "perf_metric": "none",
  "hard_fails": hard_list,
  "hard_fail_reasons": hard_list,
  "tests": {"passed": i(os.environ.get("WRO_PASSED")), "total": i(os.environ.get("WRO_TOTAL"))},
  "gates": {"scope_ok": os.environ.get("WRO_SCOPE")=="true",
            "import_origin_ok": os.environ.get("WRO_IMP")=="true",
            "correctness_ok": (os.environ.get("WRO_SCOPE")=="true"
                               and os.environ.get("WRO_IMP")=="true"
                               and not hard_list)}}
os.makedirs("/logs/verifier", exist_ok=True)
with open("/logs/verifier/reward.json", "w") as fh:
    json.dump(verdict, fh)
print(json.dumps(verdict))
PY
