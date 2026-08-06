#!/usr/bin/env bash
# Verifier for flash-online-softmax-prefill (causal scaled-dot-product attention).
# Pattern A, ACCELERATION lane, PEAK-MEMORY on H20 (naive materializes the (S,S)
# matrix -> peak ~ O(B*H*S*S); a working-set candidate stays ~ O(B*H*S*D)).
# Modes: candidate|noop|oracle|negative
set -uo pipefail
# Never trust the exec context -- the harness may exec this in a
# NON-login shell where bare python3 has no torch. Pin the venv first.
export PATH=/opt/kernelbench-venv/bin:$PATH
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/loop/dev
mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
CAND_FILE="$REPO/causal_attention.py"
export KB_CANDIDATE_MODULE="$CAND_FILE"
export KB_BASELINE_MODULE=/opt/verifier-baseline/causal_attention.py

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; trusted_restore_ok=true; hidden_correctness_ok=true
baseline_ok=true; benchmark_ok=true; anti_cheat_ok=true
expected_cases=12; actual_cases=0
speedup=1.0; ref_speedup=1.0; hw="unknown"

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

# anti-cheat: only causal_attention.py may change
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/anti_cheat_changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    causal_attention.py) : ;;
    *__pycache__*) : ;;
    *) anti_cheat_ok=false; add_hard "forbidden_edit_path:$f" ;;
  esac
done < "$LOG/anti_cheat_changed_files.txt"

# anti-cheat: framework fused attention primitives are forbidden in the candidate
# file (the attention must be built explicitly). The candidate symbol is
# `causal_attention`, so these patterns never self-match.
if [ -f "$CAND_FILE" ] && grep -Eq "scaled_dot_product_attention|_scaled_dot_product|flash_attn|torch\.nn\.MultiheadAttention|nn\.MultiheadAttention" "$CAND_FILE"; then
  anti_cheat_ok=false; add_hard "forbidden_fused_attention_primitive"
fi

# trusted restore: hidden suite ships in tests/ payload (fresh at scoring)
for f in test_causal_attention.py gpu_peakmem.py kb_attn_harness.py; do
  [ -f "$TESTS_DIR/$f" ] || { trusted_restore_ok=false; add_hard "hidden_supplement_missing:$f"; }
done

# ref_speedup: prefer the tests/-local override (uploaded fresh -> recalibrate without rebuild)
if [ -f "$TESTS_DIR/ref_speedup.txt" ]; then
  ref_speedup=$(tr -dc '0-9.' < "$TESTS_DIR/ref_speedup.txt")
elif [ -f /opt/verifier-correctness-manifest.json ]; then
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)
fi
hw=$(python3 -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')" 2>/dev/null || echo unknown)

# correctness gate (needs CUDA)
if [ "$correctness_ok" = true ]; then
  ( cd "$TESTS_DIR" && python3 test_causal_attention.py ) > "$LOG/correctness.log" 2>&1
  crc=$?
  actual_cases=$(grep -cE '^CASE_PASS ' "$LOG/correctness.log" 2>/dev/null); actual_cases=${actual_cases:-0}
  if [ "$crc" -ne 0 ] || [ "$actual_cases" -lt "$expected_cases" ]; then
    hidden_correctness_ok=false; correctness_ok=false; add_hard "hidden_correctness_failed"
  fi
fi

# peak-memory metric: baseline_peak/candidate_peak, geomean over shapes (only if correct)
if [ "$correctness_ok" = true ] && [ -f "$KB_BASELINE_MODULE" ]; then
  : > "$LOG/bench.log"
  line=$( cd "$TESTS_DIR" && python3 gpu_peakmem.py 2>>"$LOG/bench.log" )
  echo "$line" >> "$LOG/bench.log"
  speedup=$(echo "$line" | sed -n 's/.*speedup=\([0-9.-]*\).*/\1/p')
  [ -z "$speedup" ] && speedup=$(grep -oE 'speedup=[0-9.-]+' "$LOG/bench.log" | tail -1 | sed 's/speedup=//')
  speedup=${speedup:--1}
  if python3 -c "import sys;sys.exit(0 if float('$speedup')<=0 else 1)"; then
    benchmark_ok=false; add_hard "benchmark_execution_failed"
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
 "work_evidence_notes":["causal attention suite $actual_cases/$expected_cases ok","peak-memory speedup=$speedup on $hw","correctness_tail:${DIAG_TAIL}"]}
JSON
cat > "$LOG/correctness_results.json" <<JSON
{"schema_version":"kernelbench_correctness_results_v1",
 "suites":[{"name":"hidden-causal-attention-suite","visibility_class":"hidden_supplement","expected_case_count":$expected_cases,"actual_case_count":$actual_cases,"passed":$actual_cases,"failed":$((expected_cases-actual_cases)),"hard_fail_reasons":$HARD_JSON}],
 "correctness_ok":$correctness_ok,"trusted_restore_ok":$trusted_restore_ok,"hidden_correctness_ok":$hidden_correctness_ok}
JSON
cat > "$LOG/benchmark_results.json" <<JSON
{"schema_version":"kernelbench_benchmark_results_v1","task_kind":"acceleration","metric":"peak_memory_bytes",
 "direction":"higher_is_better",
 "workloads":[{"name":"causal-attention-peakmem-geomean-4shapes","speedup":$speedup,"completion_counters":{"cases":"$actual_cases/$expected_cases"}}],
 "aggregate_speedup":$speedup,"ref_speedup":$ref_speedup,"actual_hardware_type":"$hw"}
JSON
python3 "$TESTS_DIR/compute_reward.py" 2>/dev/null || python3 /tests/compute_reward.py 2>/dev/null || true
REW=$(tr -d '[:space:]' < "$LOG/reward.txt" 2>/dev/null)
echo "MODE=$MODE cases=$actual_cases/$expected_cases speedup=$speedup ref=$ref_speedup hw=$hw reward=$REW"
[ -n "$REW" ] && exit 0 || exit 1
