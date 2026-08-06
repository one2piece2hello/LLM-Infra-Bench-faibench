#!/usr/bin/env bash
# ============================================================================
# loop16 finalizer — score_engine.sh  (REUSABLE, task-agnostic)  0700 root
# ============================================================================
# Invoked ONLY by /opt/loop/submit.sh. Scores the LIVE /app/repo by running the
# task's OWN verifier in candidate mode against the solver's in-scope edits, then
# normalizes its standard JSON verdict into the dev files submit.sh/sanitize read.
#
# The task verifier (tests/test.sh, baked into /opt/loop/private/tests/) already
# performs the whole scoring contract in candidate mode: scope-diff gate,
# import-origin assert, exact-correctness gate vs the baked golden, and paired
# candidate-vs-baseline timing -> a single JSON line:
#   {"mode","reward","speedup","baseline_ms","candidate_ms","ref_speedup",
#    "metadata",{...},"hard_fails":[...],
#    "gates":{"scope_ok","import_origin_ok","correctness_ok","benchmark_ok"}}
# reward = speedup when there are no hard fails and correctness holds, else 0.
#
# This engine is generic across every task: the only per-task content is the
# baked verifier itself (test.sh + workload.py + compute_reward.py), staged into
# /opt/loop/private/tests/ at image build time.
export PATH=/opt/kernelbench-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -uo pipefail
git config --global --add safe.directory '*' 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_OUT="${LOOP_DEV_OUT:-/logs/loop/dev}"; export LOOP_DEV_OUT="$DEV_OUT"; mkdir -p "$DEV_OUT"
TESTS="$HERE/private/tests"
rm -f "$DEV_OUT/harness_error.txt" "$DEV_OUT/verdict.json" 2>/dev/null || true

if [ ! -f "$TESTS/test.sh" ]; then
  echo "baked verifier missing at $TESTS/test.sh" > "$DEV_OUT/harness_error.txt"
  exit 3
fi

# The loop16 image bakes a committed /app/repo/.git at the degraded baseline (built
# offline; its files carry a different owner/mtime than the image working tree). Refresh
# the index so the task's git scope/timing gates compare CONTENT, not stale stat — a real
# in-scope edit still shows as modified; only content-identical stat noise is cleared.
git -C /app/repo update-index -q --refresh >/dev/null 2>&1 || true

# Run the task's verifier in candidate mode against the live /app/repo.
KERNELBENCH_VERIFY_MODE=candidate bash "$TESTS/test.sh" > "$DEV_OUT/verdict.raw" 2>"$DEV_OUT/verdict.err"
vr=$?

# Normalize the verdict JSON -> dev files. A genuine harness failure (no
# parseable verdict) is reported as harness_error (submit.sh refunds the budget).
python3 - "$DEV_OUT" "$vr" <<'PY'
import json, sys
from pathlib import Path
DEV = Path(sys.argv[1]); vr = sys.argv[2]
raw = ""
try: raw = (DEV / "verdict.raw").read_text(encoding="utf-8", errors="replace")
except Exception: raw = ""

# grab the LAST line that parses as a verdict (has a "gates" key)
verdict = None
for line in reversed(raw.splitlines()):
    line = line.strip()
    if not line or not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if isinstance(obj, dict) and "gates" in obj and "hard_fails" in obj:
        verdict = obj; break

if verdict is None:
    err = (DEV / "verdict.err").read_text(errors="replace")[:400] if (DEV / "verdict.err").exists() else ""
    (DEV / "harness_error.txt").write_text(
        f"verifier produced no parseable verdict (exit {vr}); "
        f"stderr head: {err.splitlines()[0] if err.strip() else '(none)'}\n")
    # still write empty dev files so downstream reads don't crash
    (DEV / "verifier_state.json").write_text(json.dumps({"correctness_ok": False, "hard_fail_reasons": ["harness_error"]}))
    (DEV / "reward.json").write_text(json.dumps({"reward": 0.0, "speedup": 0.0}))
    (DEV / "benchmark_results.json").write_text(json.dumps({"aggregate_speedup": 0.0}))
    sys.exit(3)

hard = list(verdict.get("hard_fails") or [])
gates = verdict.get("gates") or {}
# A submission is check-passing iff the verifier assigned NO hard fail (which
# already covers scope / import-origin / correctness / timing failures).
correctness_ok = (len(hard) == 0) and bool(gates.get("correctness_ok", True))
reward = float(verdict.get("reward") or 0.0)
speedup = float(verdict.get("speedup") or 0.0)

(DEV / "verifier_state.json").write_text(json.dumps({
    "correctness_ok": correctness_ok,
    "hard_fail_reasons": hard,
    "gates": gates,
}, sort_keys=True))
(DEV / "reward.json").write_text(json.dumps({"reward": reward, "speedup": speedup}, sort_keys=True))
(DEV / "benchmark_results.json").write_text(json.dumps({"aggregate_speedup": speedup}, sort_keys=True))
sys.exit(0)
PY
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ] && [ ! -f "$DEV_OUT/harness_error.txt" ]; then
  { echo "score_engine_exit_${rc}"; tail -n 20 "$DEV_OUT/verdict.err" 2>/dev/null; } > "$DEV_OUT/harness_error.txt"
fi
exit "$rc"
