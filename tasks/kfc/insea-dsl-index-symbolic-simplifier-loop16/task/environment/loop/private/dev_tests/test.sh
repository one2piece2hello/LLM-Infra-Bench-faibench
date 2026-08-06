#!/usr/bin/env bash
# Verifier for insea-dsl-index-symbolic-simplifier-loop16 (bounded integer index-expression
# canonical simplifier). Pattern A, ACCELERATION lane, CPU deterministic proxy =
# node_count. speedup = baseline_nodecount / candidate_nodecount. No timing, no GPU:
# the value axis is the number of AST nodes in the simplified output, gated by an
# exhaustive small-range value-equivalence check (pure-Python stdlib).
# Modes: candidate|noop|oracle|negative
set -uo pipefail
# The harness may exec this in a NON-login shell; pin PATH so
# python3 resolves regardless of the base's default profile.
export PATH=/opt/kernelbench-venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/loop/dev
mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
CAND_FILE="$REPO/index_expr_simplify.py"
export KB_CANDIDATE_MODULE="$CAND_FILE"
export KB_BASELINE_MODULE=/opt/verifier-baseline/index_expr_simplify.py

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; trusted_restore_ok=true; hidden_correctness_ok=true
baseline_ok=true; benchmark_ok=true; anti_cheat_ok=true
expected_cases=12; actual_cases=0
speedup=1.0; ref_speedup=1.0; hw="cpu"; cand_nodes=0; base_nodes=0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }

if [ "$MODE" = "oracle" ] && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ]; then
  # reviewer-only mode: reset the working tree to the frozen baseline so several
  # modes running back-to-back in one container session each start clean. candidate/noop
  # NEVER resets (this branch is skipped for them), so a solver session is untouched.
  git -C "$REPO" checkout -- . 2>/dev/null || true; git -C "$REPO" clean -fdq 2>/dev/null || true
  git -C "$REPO" apply "$KERNELBENCH_ORACLE_PATCH" 2>"$LOG/oracle_apply.log" || add_hard "oracle_apply_failed"
fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" checkout -- . 2>/dev/null || true; git -C "$REPO" clean -fdq 2>/dev/null || true
  git -C "$REPO" apply "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true
fi

# anti-cheat: only index_expr_simplify.py may change
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/anti_cheat_changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    index_expr_simplify.py) : ;;
    *__pycache__*) : ;;
    *) anti_cheat_ok=false; add_hard "forbidden_edit_path:$f" ;;
  esac
done < "$LOG/anti_cheat_changed_files.txt"

# anti-cheat: the rewrite must be implemented explicitly. Forbid computer-algebra
# / symbolic-math packages and the upstream source library.
if [ -f "$CAND_FILE" ] && grep -Eq "sympy|symengine|diofant|sage\.|PatternMatcher|UPat|tinygrad" "$CAND_FILE"; then
  anti_cheat_ok=false; add_hard "forbidden_symbolic_library_primitive"
fi

# trusted restore: hidden suite ships in tests/ payload (fresh at scoring)
for f in test_index_expr_simplify.py node_bench.py kb_symbolic_harness.py; do
  [ -f "$TESTS_DIR/$f" ] || { trusted_restore_ok=false; add_hard "hidden_supplement_missing:$f"; }
done

# ref_speedup: prefer the tests/-local override (uploaded fresh -> recalibrate without rebuild)
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi
hw="cpu:$(uname -m 2>/dev/null || echo unknown)"

# correctness gate (pure python; no GPU)
if [ "$correctness_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 test_index_expr_simplify.py ) > "$LOG/correctness.log" 2>&1
  crc=$?
  actual_cases=$(grep -cE '^CASE_PASS ' "$LOG/correctness.log" 2>/dev/null); actual_cases=${actual_cases:-0}
  if [ "$crc" -ne 0 ] || [ "$actual_cases" -lt "$expected_cases" ]; then
    hidden_correctness_ok=false; correctness_ok=false; add_hard "hidden_correctness_failed"
  fi
fi

# benchmark: (1) candidate & baseline outputs each agree with the reference by
# value-equivalence on the corpus, then (2) node_count of candidate vs frozen
# baseline; speedup = base_nodes / cand_nodes. Deterministic, no timing.
parse_nodecount(){ sed -n 's/^NODECOUNT=\([0-9]*\).*/\1/p' "$1" 2>/dev/null | head -1; }
if [ "$correctness_ok" = true ] && [ -f "$KB_BASELINE_MODULE" ]; then
  ( cd "$TESTS_DIR" && python3 node_bench.py verify ) > "$LOG/verify.log" 2>&1
  if [ $? -ne 0 ] || ! grep -q '^VERIFY_OK' "$LOG/verify.log"; then
    benchmark_ok=false; add_hard "candidate_reference_mismatch"
  fi
  if [ "$benchmark_ok" = true ]; then
    ( cd "$TESTS_DIR" && python3 node_bench.py candidate ) > "$LOG/cand.out.log" 2>"$LOG/cand.err.log"
    ( cd "$TESTS_DIR" && python3 node_bench.py baseline ) > "$LOG/base.out.log" 2>"$LOG/base.err.log"
    cand_nodes=$(parse_nodecount "$LOG/cand.out.log"); cand_nodes=${cand_nodes:-0}
    base_nodes=$(parse_nodecount "$LOG/base.out.log"); base_nodes=${base_nodes:-0}
    if [ -z "$cand_nodes" ] || [ "$cand_nodes" -le 0 ] 2>/dev/null || [ -z "$base_nodes" ] || [ "$base_nodes" -le 0 ] 2>/dev/null; then
      benchmark_ok=false; add_hard "nodecount_unparsed"
    else
      speedup=$(python3 -c "print(f'{$base_nodes/$cand_nodes:.5f}')")
    fi
  fi
else
  benchmark_ok=false
fi

HARD_JSON="[]"; if [ ${#HARD[@]} -gt 0 ]; then HARD_JSON=$(printf '"%s",' "${HARD[@]}"); HARD_JSON="[${HARD_JSON%,}]"; fi
DIAG_TAIL=""
if [ "$correctness_ok" = false ] && [ -f "$LOG/correctness.log" ]; then
  DIAG_TAIL=$(tail -60 "$LOG/correctness.log" | sed 's/"/\\"/g' | tr '\n' '\t' | sed 's/\t/\\n/g')
fi
cat > "$LOG/verifier_state.json" <<JSON
{"schema_version":"kernelbench_verifier_state_v1","task_kind":"acceleration",
 "correctness_ok":$correctness_ok,"trusted_restore_ok":$trusted_restore_ok,
 "hidden_correctness_ok":$hidden_correctness_ok,"baseline_ok":$baseline_ok,
 "benchmark_ok":$benchmark_ok,"anti_cheat_ok":$anti_cheat_ok,
 "hard_fail_reasons":$HARD_JSON,
 "expected_case_count":$expected_cases,"actual_case_count":$actual_cases,
 "expected_benchmark_count":1,"actual_benchmark_count":1,
 "actual_hardware_type":"$hw","mode":"$MODE",
 "work_evidence_notes":["simplifier suite $actual_cases/$expected_cases ok","node_count candidate=$cand_nodes baseline=$base_nodes speedup=$speedup","correctness_tail:${DIAG_TAIL}"]}
JSON
cat > "$LOG/correctness_results.json" <<JSON
{"schema_version":"kernelbench_correctness_results_v1",
 "suites":[{"name":"hidden-index-simplifier-suite","visibility_class":"hidden_supplement","expected_case_count":$expected_cases,"actual_case_count":$actual_cases,"passed":$actual_cases,"failed":$((expected_cases-actual_cases)),"hard_fail_reasons":$HARD_JSON}],
 "correctness_ok":$correctness_ok,"trusted_restore_ok":$trusted_restore_ok,"hidden_correctness_ok":$hidden_correctness_ok}
JSON
cat > "$LOG/benchmark_results.json" <<JSON
{"schema_version":"kernelbench_benchmark_results_v1","task_kind":"acceleration","metric":"node_count",
 "direction":"higher_is_better",
 "workloads":[{"name":"index-expr-recombine-nodecount","speedup":$speedup,"candidate_nodecount":$cand_nodes,"baseline_nodecount":$base_nodes,"completion_counters":{"cases":"$actual_cases/$expected_cases"}}],
 "aggregate_speedup":$speedup,"ref_speedup":$ref_speedup,"actual_hardware_type":"$hw"}
JSON
python3 "$TESTS_DIR/compute_reward.py" 2>/dev/null || python3 /tests/compute_reward.py 2>/dev/null || true
REW=$(tr -d '[:space:]' < "$LOG/reward.txt" 2>/dev/null)
echo "MODE=$MODE cases=$actual_cases/$expected_cases cand_nodes=$cand_nodes base_nodes=$base_nodes speedup=$speedup ref=$ref_speedup hw=$hw reward=$REW"
[ -n "$REW" ] && exit 0 || exit 1
