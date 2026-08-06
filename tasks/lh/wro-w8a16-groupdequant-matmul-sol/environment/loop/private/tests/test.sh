#!/usr/bin/env bash
# Verifier for wro-w8a16-groupdequant-matmul-sol — Type-2 Long-horizon, B2 BEAT (acceleration), GPU H20.
# Subsystem = a group-quantised int8-weight / fp16-activation matmul (w8a16.w8a16_matmul).
# Scope = w8a16/matmul.py. Returns a @ dequant(qweight, scales, zeros, group_size):
# a[M,K] fp16, qweight[K,N] int8 (asymmetric group-quant), scales/zeros [K//g, N] (fp16/int8),
# fp32 accumulate, fp16 [M,N] result. Baked (noop) = a correct-but-SLOW baseline that expands
# the per-group scale/zero grids to full [K,N], materialises the dense fp32 weight in global
# memory, and runs a dense matmul on it (several extra HBM passes). The solver must stream the
# packed int8 weight and dequantise on the fly (reloading per-group scale/zero) in its own kernel
# to beat the oracle. reward = gated_oracle; perf_metric = speedup (SOL-anchored, memory-bound).
# Modes: candidate | noop | oracle | baseline2 | negative.
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"   # non-login exec has no torch on bare python3
export PYTHONPATH="/app/repo:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"; export WRO_LOG="$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("w8a16/matmul.py")
# banned tokens: a submitted scope file routing to a prebuilt/library matmul instead of
# computing the product in its own kernel (W17/D7). DIFF-BASED (only ADDED lines count).
# Comments count. Dequantising in registers + a hand-written block matmul is NOT banned.
BANNED=("torch.matmul" "torch.mm" "torch.bmm" "F.linear" "torch.nn.functional.linear" \
        "torch.einsum" "cublas" "cutlass" "addmm" ".matmul(")

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; import_origin_ok=true; benchmark_ok=true; ban_ok=true
corr_frac=0.0; base_ms=-1; cand_ms=-1; ref_speedup=1.0

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; }

# ---- apply oracle / baseline2 / negative patch by mode (candidate & noop: no patch) ----
if [ "$MODE" = "oracle" ]    && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];    then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"    2>"$LOG/oracle_apply.log"    || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "baseline2" ] && [ -n "${KERNELBENCH_BASELINE2_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_BASELINE2_PATCH" 2>"$LOG/baseline2_apply.log" || add_hard "baseline2_apply_failed"; fi
if [ "$MODE" = "negative" ]  && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ];  then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH"  2>"$LOG/negative_apply.log"  || true; fi

# ---- (1) scope-diff HARD GATE: only the scope file(s) may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r ff; do
  [ -z "$ff" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$ff" = "$s" ] && keep=true; done
  case "$ff" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$ff"; }
done < "$LOG/changed_files.txt"

# ---- (2) vendor-op ban: candidate's ADDED lines must not route to a library matmul ----
for s in "${SCOPE[@]}"; do
  [ -f "$REPO/$s" ] || continue
  git -C "$REPO" diff HEAD -- "$s" 2>/dev/null | grep '^+' | grep -v '^+++' > "$LOG/added_lines.txt" || true
  for tok in "${BANNED[@]}"; do
    if grep -Fq "$tok" "$LOG/added_lines.txt"; then ban_ok=false; add_hard "vendor_op_ban:$tok"; fi
  done
done

# ---- (3) import-origin assert: w8a16 under test must be the baked /app/repo tree ----
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import w8a16, os, sys
loc = os.path.dirname(w8a16.__file__)
print("PKG_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (4) trusted-restore: harness uploaded fresh (never baked) ----
for fn in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$fn" ] || { add_hard "hidden_supplement_missing:$fn"; }
done

# ---- ref_speedup (oracle-calibrated, baked at build; draft pre-bake) ----
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

# ---- (5) CORRECTNESS GATE (graded): scope output must match the fp32 reference over the suite ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ] && [ "$ban_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_frac=$(grep WRO_W8A16_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_W8A16_RESULT ',1)[1]).get('correctness_frac',0.0))
except Exception: print('0.0')" 2>/dev/null || echo 0.0)
fi

# ---- (6) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) — only if fully correct ----
if [ "$(python3 -c "print(1 if $corr_frac>=1.0 else 0)" 2>/dev/null || echo 0)" = "1" ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_W8A16_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_W8A16_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  cp "$LOG/cand_timing.out" "$LOG/cand_timing_sol.out" 2>/dev/null || true
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_W8A16_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_W8A16_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ] || { benchmark_ok=false; add_hard "timing_invalid"; }
fi

# ---- delegate to compute_reward.py (gated_oracle) ----
export WRO_MODE="$MODE" WRO_HARD="${HARD[*]:-}" WRO_CORR_FRAC="$corr_frac" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" WRO_BENCH="$benchmark_ok" WRO_BAN="$ban_ok"
python3 "$TESTS_DIR/compute_reward.py"
