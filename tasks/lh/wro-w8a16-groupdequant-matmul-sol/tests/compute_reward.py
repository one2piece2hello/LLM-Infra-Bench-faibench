#!/usr/bin/env python3
"""compute_reward.py — reward.md-formula writer for wro-w8a16-groupdequant-matmul-sol.

Reads the environment populated by test.sh and emits the 5-file verifier output
contract under /logs/verifier. Reward = the reward.md PERFORMANCE formula:

  reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

  speedup = baseline_ms / candidate_ms (candidate vs. its own frozen baseline)
  ref_speedup = the oracle's own median speedup in the same image

Pre-gates (any hit -> reward = 0.0, computed in this precedence order):
  1. hard fail already raised upstream by test.sh (build/import/scope/vendor-op-ban/etc)
  2. correctness_frac < 1.0 (ANY hidden case fails -> hard zero; no partial credit)
  3. speedup <= 1.0 (candidate did not beat its own baseline)
  4. ref_speedup <= 1.0 (reference solution invalid)

NOTE (2026-07 reform): the old "vs_oracle > 2.0 -> hard_fail implausible_vs_oracle"
plausibility guard has been REMOVED (same cluster-wide guard as wro-triton-dqgemm-sol). It
was not anti-cheat-load-bearing (anti-cheat lives in the separate scope_ok / import_origin_ok
/ vendor_op_ban_ok gates, all still enforced by test.sh above); it only ever hard-zeroed
legitimate high-headroom wins, and the new min(1.0, ...) cap already bounds reward <= 1.0
without needing a hand-rolled ceiling. Verified safe to remove: oracle.patch / baseline2.patch
/ negative.patch all replace the torch.matmul baseline with the same kernel family, none
depend on this guard firing.
Dual-emit: also surface raw `speedup` so the stock loop16 reward_moves bar (raw >2x
swing) fires while scoring stays keyed to the gated `reward`.
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
ban_ok = os.environ.get("WRO_BAN") == "true"

speedup = (base_ms / cand_ms) if (base_ms > 0 and cand_ms > 0) else 0.0
vs_oracle = (speedup / ref_speedup) if ref_speedup > 0 else 0.0

# reward.md performance pre-gates, in precedence order (each hit -> reward = 0.0)
if hard:
    score = 0.0
elif corr_frac < 1.0:
    score = 0.0
    hard = hard + ["correctness_failed"]
elif speedup <= 1.0:
    score = 0.0
    hard = hard + ["no_speedup"]
elif ref_speedup <= 1.0:
    score = 0.0
    hard = hard + ["invalid_ref_speedup"]
else:
    score = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))

verifier_state = {
    "mode": mode,
    "hard_fails": hard,
    "gates": {
        "scope_ok": scope_ok,
        "import_origin_ok": import_ok,
        "benchmark_ok": bench_ok,
        "vendor_op_ban_ok": ban_ok,
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
    # reward.md result-JSON schema (performance):
    "task_type": "performance",
    "reward": round(score, 6),
    "hard_fail_reasons": hard,
    "speedup": round(speedup, 6),
    "ref_speedup": ref_speedup,
    "cv": {"baseline": None, "candidate": None},  # test.sh timing is single-shot, not ABBA>=5; not measured
    # legacy/back-compat fields (score_engine + loop16 smoke still read these):
    "score": round(score, 6),
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
