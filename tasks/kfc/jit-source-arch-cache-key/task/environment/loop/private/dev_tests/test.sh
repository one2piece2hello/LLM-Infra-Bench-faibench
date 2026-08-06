#!/usr/bin/env bash
# Verifier for jit-source-arch-cache-key (stable identity for a build-spec).
# Pattern A, ACCELERATION lane. Deterministic value axis = recompile_count = number of
# DISTINCT identities produced over a fixed labeled workload of build-specs. speedup =
# baseline_recompiles / candidate_recompiles (= baseline_distinct_identities /
# candidate_distinct_identities). Pure-Python stdlib -> a plain count, no profiler
# needed. A false merge (two different-class specs sharing an identity -- e.g. two specs
# differing only in target profile) is a hard correctness failure -> reward 0.
# Modes: candidate|noop|oracle|negative
set -uo pipefail
# The harness may exec this in a NON-login shell; pin PATH so python3
# resolves regardless of the base's default profile.
export PATH=/opt/kernelbench-venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/loop/dev
mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
CAND_FILE="$REPO/identity_key.py"
export KB_CANDIDATE_MODULE="$CAND_FILE"
export KB_BASELINE_MODULE=/opt/verifier-baseline/identity_key.py

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; trusted_restore_ok=true; hidden_correctness_ok=true
baseline_ok=true; benchmark_ok=true; anti_cheat_ok=true
expected_cases=12; actual_cases=0
speedup=1.0; ref_speedup=1.0; hw="cpu"; cand_keys=0; base_keys=0

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

# anti-cheat: only identity_key.py may change
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/anti_cheat_changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    identity_key.py) : ;;
    *__pycache__*) : ;;
    *) anti_cheat_ok=false; add_hard "forbidden_edit_path:$f" ;;
  esac
done < "$LOG/anti_cheat_changed_files.txt"

# anti-cheat: the normalization must be implemented in plain Python. Forbid pulling
# in a tensor/graph framework or reading the hidden workload/reference harness.
if [ -f "$CAND_FILE" ] && grep -Eq "cupy|jax|jaxlib|torch|triton|tensorflow|kb_identity_harness|test_identity_key|import[[:space:]]+tests|from[[:space:]]+tests" "$CAND_FILE"; then
  anti_cheat_ok=false; add_hard "forbidden_framework_or_harness_reference"
fi

# trusted restore: hidden suite ships in tests/ payload (fresh at scoring)
for f in test_identity_key.py count_bench.py kb_identity_harness.py; do
  [ -f "$TESTS_DIR/$f" ] || { trusted_restore_ok=false; add_hard "hidden_supplement_missing:$f"; }
done

# ref_speedup: prefer the tests/-local override (uploaded fresh -> recalibrate without rebuild)
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi
hw="cpu:$(uname -m 2>/dev/null || echo unknown)"

# correctness gate (pure python; no GPU): stable string + no false merge properties.
if [ "$correctness_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 test_identity_key.py ) > "$LOG/correctness.log" 2>&1
  crc=$?
  actual_cases=$(grep -cE '^CASE_PASS ' "$LOG/correctness.log" 2>/dev/null); actual_cases=${actual_cases:-0}
  if [ "$crc" -ne 0 ] || [ "$actual_cases" -lt "$expected_cases" ]; then
    hidden_correctness_ok=false; correctness_ok=false; add_hard "hidden_correctness_failed"
  fi
fi

# benchmark: (1) confirm the candidate never gives two different-class signatures the
# same identity (safety), then (2) count distinct identities for candidate vs the
# frozen baseline over the fixed workload; speedup = base_keys / cand_keys.
parse_count(){ sed -n 's/^COUNT=//p' "$1" 2>/dev/null | head -1; }
if [ "$correctness_ok" = true ] && [ -f "$KB_BASELINE_MODULE" ]; then
  ( cd "$TESTS_DIR" && python3 count_bench.py verify ) > "$LOG/verify.log" 2>&1
  if [ $? -ne 0 ] || ! grep -q '^VERIFY_OK' "$LOG/verify.log"; then
    benchmark_ok=false; add_hard "candidate_false_merge"
  fi
  if [ "$benchmark_ok" = true ]; then
    ( cd "$TESTS_DIR" && python3 count_bench.py candidate ) > "$LOG/cand.out.log" 2>&1
    ( cd "$TESTS_DIR" && python3 count_bench.py baseline ) > "$LOG/base.out.log" 2>&1
    cand_keys=$(parse_count "$LOG/cand.out.log"); cand_keys=${cand_keys:-0}
    base_keys=$(parse_count "$LOG/base.out.log"); base_keys=${base_keys:-0}
    if [ -z "$cand_keys" ] || [ "$cand_keys" -le 0 ] 2>/dev/null || [ -z "$base_keys" ] || [ "$base_keys" -le 0 ] 2>/dev/null; then
      benchmark_ok=false; add_hard "distinct_identity_count_unparsed"
    else
      speedup=$(python3 -c "print(f'{$base_keys/$cand_keys:.5f}')")
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
 "work_evidence_notes":["identity suite $actual_cases/$expected_cases ok","distinct identities candidate=$cand_keys baseline=$base_keys speedup=$speedup","correctness_tail:${DIAG_TAIL}"]}
JSON
cat > "$LOG/correctness_results.json" <<JSON
{"schema_version":"kernelbench_correctness_results_v1",
 "suites":[{"name":"hidden-identity-suite","visibility_class":"hidden_supplement","expected_case_count":$expected_cases,"actual_case_count":$actual_cases,"passed":$actual_cases,"failed":$((expected_cases-actual_cases)),"hard_fail_reasons":$HARD_JSON}],
 "correctness_ok":$correctness_ok,"trusted_restore_ok":$trusted_restore_ok,"hidden_correctness_ok":$hidden_correctness_ok}
JSON
cat > "$LOG/benchmark_results.json" <<JSON
{"schema_version":"kernelbench_benchmark_results_v1","task_kind":"acceleration","metric":"recompile_count",
 "direction":"higher_is_better",
 "workloads":[{"name":"buildspec-identity-count","speedup":$speedup,"candidate_recompiles":$cand_keys,"baseline_recompiles":$base_keys,"completion_counters":{"cases":"$actual_cases/$expected_cases"}}],
 "aggregate_speedup":$speedup,"ref_speedup":$ref_speedup,"actual_hardware_type":"$hw"}
JSON
python3 "$TESTS_DIR/compute_reward.py" 2>/dev/null || python3 /tests/compute_reward.py 2>/dev/null || true
REW=$(tr -d '[:space:]' < "$LOG/reward.txt" 2>/dev/null)
echo "MODE=$MODE cases=$actual_cases/$expected_cases cand_recompiles=$cand_keys base_recompiles=$base_keys speedup=$speedup ref=$ref_speedup hw=$hw reward=$REW"
[ -n "$REW" ] && exit 0 || exit 1
