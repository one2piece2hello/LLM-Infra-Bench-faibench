#!/usr/bin/env bash
# Verifier for wro-flashattn-cute-blocksparse — Type-2 Long-horizon, B2 BEAT (acceleration).
# Subsystem = BLOCK-SPARSE multi-head attention FORWARD (scaled dot-product where the caller
# partitions Q along the query axis and K/V along the key axis into fixed-size blocks and
# passes explicit ordered integer index arrays that name which (M-block, KV-block) pairs
# are allowed; supports grouped-query attention and cross-attention). Scope =
# flash_attn/cute/interface.py (public flash_attn_func entry threading
# block_sparse_tensors=BlockSparseTensorsTorch(mask_block_cnt, mask_block_idx,
# full_block_cnt, full_block_idx, block_size) through). Baseline (baked, noop) = a
# correct-but-SLOW eager attention that reads the passed block_sparse_tensors, rebuilds a
# dense boolean allow-mask from the index arrays, and applies it to the full O(seqlen^2)
# score matrix; the solver restores the fused single-pass path (skipping whole
# (m_block, kv_block) pairs at scheduler level via block_sparse_utils.produce_block_sparse_loads)
# to beat the oracle. Memory-bound: sparsity gives multiplicative long-seq headroom.
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
export PYTHONPATH="/app/repo:/app/stubs:${PYTHONPATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"; export WRO_LOG="$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("flash_attn/cute/interface.py")
# Ban pulling in a prebuilt/compiled/external fused attention instead of implementing the fast path
# within the CuTe subsystem (diff-based; comments count). None occur in the oracle (which uses the
# repo's own FlashAttnFunc CuTe kernel).
BANNED=("scaled_dot_product_attention" "flash_attn_2_cuda" "flash_attn_3_cuda" "xformers" \
        "import flash_attn_interface" "from flash_attn.flash_attn_interface" "vllm" "sageattention")

HARD=(); add_hard(){ HARD+=("$1"); }
scope_ok=true; import_origin_ok=true; benchmark_ok=true; ban_ok=true
corr_frac=0.0; base_ms=-1; cand_ms=-1; ref_speedup=1.0

cd "$REPO" 2>/dev/null || add_hard "repo_missing"

if [ "$MODE" = "oracle" ]   && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];   then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"   2>"$LOG/oracle_apply.log"   || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true; fi

# (1) scope-diff HARD GATE
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r ff; do
  [ -z "$ff" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$ff" = "$s" ] && keep=true; done
  case "$ff" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$ff"; }
done < "$LOG/changed_files.txt"

# (2) ban scan — DIFF-BASED (only solver-ADDED lines)
for s in "${SCOPE[@]}"; do
  [ -f "$REPO/$s" ] || continue
  git -C "$REPO" diff HEAD -- "$s" 2>/dev/null | grep '^+' | grep -v '^+++' > "$LOG/added_lines.txt" || true
  for tok in "${BANNED[@]}"; do
    grep -Fq "$tok" "$LOG/added_lines.txt" && { ban_ok=false; add_hard "vendor_op_ban:$tok"; }
  done
done

# (3) import-origin assert (scope module resolves under /app/repo, not a site-packages twin)
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import os, sys
import flash_attn.cute.interface as m
loc = os.path.dirname(m.__file__)
print("FA_CUTE_LOC", loc)
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# (4) trusted-restore of hidden supplements
for fn in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$fn" ] || add_hard "hidden_supplement_missing:$fn"
done

# ref_speedup (baked calibration manifest)
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

# (5) CORRECTNESS (pass-rate over many hidden cases)
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ] && [ "$ban_ok" = true ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  corr_frac=$(grep WRO_FA_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WRO_FA_RESULT ',1)[1]).get('correctness_frac',0.0))
except Exception: print('0.0')" 2>/dev/null || echo 0.0)
fi

# (6) TIMING (only if fully correct) — candidate vs baked degraded baseline
if [ "$(python3 -c "print(1 if $corr_frac>=1.0 else 0)" 2>/dev/null || echo 0)" = "1" ] && [ "${#HARD[@]}" -eq 0 ]; then
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WRO_FA_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_FA_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$TESTS_DIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WRO_FA_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WRO_FA_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ] || { benchmark_ok=false; add_hard "timing_invalid"; }
fi

export WRO_MODE="$MODE" WRO_HARD="${HARD[*]:-}" WRO_CORR_FRAC="$corr_frac" \
       WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" WRO_REF="$ref_speedup" \
       WRO_SCOPE="$scope_ok" WRO_IMP="$import_origin_ok" WRO_BENCH="$benchmark_ok" WRO_BAN="$ban_ok"
python3 "$TESTS_DIR/compute_reward.py"
