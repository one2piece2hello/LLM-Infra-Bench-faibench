#!/usr/bin/env bash
# Verifier for wli-fla-retention-chunkscan-loop16.
# Subsystem: multi-scale retention (RetNet) chunked linear attention.
# Scope = fla/ops/retention/chunk.py. Type-2 Long-horizon, PERFORMANCE family (B2 beat).
# Modes: candidate | noop | oracle | negative.  reward = raw speedup (base_ms/cand_ms), noop~=1.0.
# baseline (timing denominator) = the frozen degraded tree at HEAD (correct-but-slow eager recurrence).
set -uo pipefail
export PATH="/opt/kernelbench-venv/bin:/opt/venv/bin:/usr/local/bin:$PATH"
# --- Block the optional `tilelang` backend BEFORE any `import fla`. The base image ships
# nvidia_cutlass_dsl (a TVM-FFI DSL); fla's optional backends also try to `import tilelang`
# (TVM-based) -> the two TVM-ffi runtimes double-register a type -> an UNCATCHABLE C++ abort
# ("tvm::ffi::Error ... already registered ... Aborted (core dumped)", exit 134) on `import fla`.
# A sitecustomize on PYTHONPATH auto-runs at every python3 startup (import-origin heredoc AND
# workload.py) and raises ModuleNotFoundError(name='tilelang') — which is exactly what fla's
# optional-import guard swallows, so fla loads and falls back to its Triton chunk path (the
# oracle path does NOT need tilelang). No image rebuild: tests are uploaded fresh at scoring.
WRO_SITE="$(mktemp -d)"
cat > "$WRO_SITE/sitecustomize.py" <<'PYEOF'
import sys, importlib.abc
class _TLBlock(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.', 1)[0] == 'tilelang':
            raise ModuleNotFoundError("tilelang disabled for kernel eval (WRO)", name='tilelang')
        return None
sys.meta_path.insert(0, _TLBlock())
PYEOF
export PYTHONPATH="$WRO_SITE:/app/repo:${PYTHONPATH:-}"   # fla resolves to the baked degraded tree
git config --global --add safe.directory '*' 2>/dev/null || true
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("fla/ops/retention/chunk.py")

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

# ---- (1) scope-diff HARD GATE: only scope files may change in /app/repo ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  keep=false
  for s in "${SCOPE[@]}"; do [ "$f" = "$s" ] && keep=true; done
  case "$f" in *__pycache__*|*.pyc) keep=true;; esac
  [ "$keep" = true ] || { scope_ok=false; add_hard "out_of_scope_edit:$f"; }
done < "$LOG/changed_files.txt"

# ---- (2) import-origin assert: fla under test must be the baked /app/repo tree ----
# NOTE: run with cwd=/app/repo and sys.path[0]='' (python3 -), so `import fla` resolves
# relative to cwd and fla.__file__ can be the RELATIVE path 'fla/__init__.py'. Use abspath
# (resolves against cwd=/app/repo) so the assert reflects the true on-disk origin — the base
# ships NO fla in site-packages, so the only fla is the baked /app/repo tree.
python3 - > "$LOG/import_origin.log" 2>&1 <<'PY'
import fla, os, sys
loc = os.path.abspath(fla.__file__)
print("FLA_LOC", loc, "CWD", os.getcwd())
sys.exit(0 if loc.startswith("/app/repo") else 3)
PY
[ $? -eq 0 ] || { import_origin_ok=false; add_hard "import_origin_not_app_repo"; }

# ---- (3) trusted-restore: harness uploaded fresh (never baked) ----
for f in workload.py compute_reward.py; do
  [ -f "$TESTS_DIR/$f" ] || { add_hard "hidden_supplement_missing:$f"; correctness_ok=false; }
done

# ---- ref_speedup (oracle-calibrated; metadata only, never the reward) ----
[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

# ---- (4) CORRECTNESS GATE: scope output must match the independent fp32 reference ----
if [ "$scope_ok" = true ] && [ "$import_origin_ok" = true ]; then
  ( cd "$RUNDIR" && python3 workload.py correctness ) > "$LOG/correctness.out" 2>&1
  cok=$(grep WLI_RET_RESULT "$LOG/correctness.out" | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read().split('WLI_RET_RESULT ',1)[1]).get('correctness_ok'))
except Exception: print('False')" 2>/dev/null || echo False)
  if [ "$cok" != "True" ]; then correctness_ok=false; add_hard "correctness_failed"; fi
fi

# ---- (5) TIMING: candidate wall + baseline (frozen degraded tree at HEAD) ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] || [ -z "${HARD[*]:-}" ]; }; then
  ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/cand_timing.out" 2>&1 || add_hard "candidate_timing_failed"
  cand_ms=$(grep WLI_RET_RESULT "$LOG/cand_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WLI_RET_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash -q 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || true
  ( cd "$RUNDIR" && python3 workload.py timing ) > "$LOG/base_timing.out" 2>&1 || add_hard "baseline_timing_failed"
  base_ms=$(grep WLI_RET_RESULT "$LOG/base_timing.out" | python3 -c "import json,sys;print(json.loads(sys.stdin.read().split('WLI_RET_RESULT ',1)[1]).get('timing_ms',-1))" 2>/dev/null || echo -1)
  git -C "$REPO" stash pop -q 2>/dev/null || true
  if [ "$(python3 -c "print(1 if $cand_ms>0 and $base_ms>0 else 0)")" = "1" ]; then
    speedup=$(python3 -c "print(round($base_ms/$cand_ms,6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

# ---- verdict ----
reward=0.0
nhard="${#HARD[@]}"
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ]; then reward="$speedup"; fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WLI_MODE="$MODE" WLI_REWARD="$reward" WLI_SPEEDUP="$speedup" \
       WLI_BASE_MS="$base_ms" WLI_CAND_MS="$cand_ms" WLI_REF="$ref_speedup" \
       WLI_HARD="$hard_str" WLI_SCOPE="$scope_ok" WLI_IMP="$import_origin_ok" \
       WLI_CORR="$correctness_ok" WLI_BENCH="$benchmark_ok"
python3 - > "$LOG/reward.json" <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
ref = f(os.environ.get("WLI_REF", "1")); sp = f(os.environ.get("WLI_SPEEDUP", "0"))
v = {
  "mode": os.environ.get("WLI_MODE"),
  "reward": f(os.environ.get("WLI_REWARD")),
  "speedup": sp,
  "baseline_ms": f(os.environ.get("WLI_BASE_MS")),
  "candidate_ms": f(os.environ.get("WLI_CAND_MS")),
  "ref_speedup": ref,
  "metadata": {"vs_oracle_ratio": (sp/ref) if ref > 0 else None},
  "hard_fails": os.environ.get("WLI_HARD", "").split(),
  "gates": {"scope_ok": os.environ.get("WLI_SCOPE") == "true",
            "import_origin_ok": os.environ.get("WLI_IMP") == "true",
            "correctness_ok": os.environ.get("WLI_CORR") == "true",
            "benchmark_ok": os.environ.get("WLI_BENCH") == "true"}}
print(json.dumps(v))
PY
cat "$LOG/reward.json"
