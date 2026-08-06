#!/usr/bin/env python3
"""compute_reward.py -- performance-type reward writer for wro-causal-delivery-vclock-coupled.

Reads the environment populated by test.sh and emits the 5-file verifier output
contract under /logs/verifier. Reward follows the canonical performance-type
formula (see the bench reward spec):

  reward = 0.0                                             if any hard_fail (front gate)
         = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0    otherwise

  speedup     = baseline_ms / candidate_ms   (candidate vs baseline, same image)
  ref_speedup = the oracle's speedup over the same baseline (baked calibration)

Front gate (ANY one hit -> reward = 0, the log formula is never evaluated):
  - correctness_frac < 1.0   (no partial credit any more -- any failing case zeroes the task)
  - speedup <= 1.0           (candidate must clear the baseline; a noop scores exactly 0)
  - ref_speedup <= 1.0       (a degenerate/invalid oracle calibration)
  - any other hard_fail already raised upstream by test.sh (scope/import/build/cheat gates)

Plausibility guard (anti-cheat backstop, NOT a score ceiling): the new formula already
self-caps at 1.0 once vs_oracle = speedup/ref_speedup reaches ref_speedup (i.e. speedup
reaches ref_speedup**2), so the OLD flat "vs_oracle > 2.0" cutoff would clip legitimate
high scores long before that cap. The guard is now anchored relative to ref_speedup and
only fires far beyond the point where the formula itself already saturates -- i.e. it
only catches results implausible enough to look like a measurement artifact / cheat,
never a genuinely strong optimization.
"""
import json
import math
import os


def f(x, d=-1.0):
    try:
        return float(x)
    except Exception:
        return d


LOG = os.environ.get("WRO_LOG", "/logs/verifier")
os.makedirs(LOG, exist_ok=True)

mode = os.environ.get("WRO_MODE", "candidate")
hard = os.environ.get("WRO_HARD", "").split()
corr_frac = f(os.environ.get("WRO_CORR_FRAC", "0"))
base_ms = f(os.environ.get("WRO_BASE_MS", "-1"))
cand_ms = f(os.environ.get("WRO_CAND_MS", "-1"))
ref_speedup = f(os.environ.get("WRO_REF", "1"), 1.0)

scope_ok = os.environ.get("WRO_SCOPE") == "true"
import_ok = os.environ.get("WRO_IMP") == "true"
bench_ok = os.environ.get("WRO_BENCH") == "true"

speedup = (base_ms / cand_ms) if (base_ms > 0 and cand_ms > 0) else 0.0
vs_oracle = (speedup / ref_speedup) if ref_speedup > 0 else 0.0

# ---- performance-type front gate (the bench reward spec #1-6); nothing below awards partial credit ----
if corr_frac < 1.0:
    hard = hard + ["correctness_incomplete"]
if speedup <= 1.0:
    hard = hard + ["speedup_not_above_baseline"]
if ref_speedup <= 1.0:
    hard = hard + ["ref_speedup_invalid"]
# plausibility guard, rescaled relative to ref_speedup so it never fires inside the region the
# min(1.0, ...) cap already governs -- only a truly implausible multiple trips it.
if not hard and vs_oracle > 10.0 * ref_speedup:
    hard = hard + ["implausible_vs_oracle:%.3f" % vs_oracle]

if hard:
    score = 0.0
else:
    score = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))

verifier_state = {
    "mode": mode,
    "hard_fails": hard,
    "gates": {
        "scope_ok": scope_ok,
        "import_origin_ok": import_ok,
        "correctness_ok": (corr_frac >= 1.0),
        "benchmark_ok": bench_ok,
    },
}
correctness_results = {"correctness_frac": corr_frac, "passed": corr_frac >= 1.0}
benchmark_results = {
    "baseline_ms": base_ms,
    "candidate_ms": cand_ms,
    "speedup": round(speedup, 6),
    "ref_speedup": ref_speedup,
    "metadata": {"vs_oracle": round(vs_oracle, 6)},
}
reward = {
    "task_type": "performance",
    "reward": round(score, 6),
    "score": round(score, 6),
    "hard_fail_reasons": hard,
    "speedup": round(speedup, 6),
    "ref_speedup": ref_speedup,
    "cv": None,
    "mode": mode,
    "correctness_frac": corr_frac,
    "vs_oracle": round(vs_oracle, 6),
    "hard_fails": hard,
    "gates": verifier_state["gates"],
}

with open(os.path.join(LOG, "verifier_state.json"), "w") as fh:
    json.dump(verifier_state, fh)
with open(os.path.join(LOG, "correctness_results.json"), "w") as fh:
    json.dump(correctness_results, fh)
with open(os.path.join(LOG, "benchmark_results.json"), "w") as fh:
    json.dump(benchmark_results, fh)
with open(os.path.join(LOG, "reward.json"), "w") as fh:
    json.dump(reward, fh)
with open(os.path.join(LOG, "reward.txt"), "w") as fh:
    fh.write(str(round(score, 6)))
# the line loop16 score_engine parses: last stdout JSON object with gates + hard_fails
print(json.dumps(reward))
