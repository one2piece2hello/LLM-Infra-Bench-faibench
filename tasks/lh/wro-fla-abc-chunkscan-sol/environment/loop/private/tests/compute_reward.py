#!/usr/bin/env python3
"""compute_reward.py — gated_oracle reward writer for wro-triton-tcgemm-sol.

Reads the environment populated by test.sh and emits the 5-file verifier output
contract under /logs/verifier. Reward = the CANONICAL gated_oracle form:

  score = 0.0                         if any hard_fail
        = 0.5 * correctness_frac      if correctness_frac < 1.0
        = 0.5 + 0.5 * vs_oracle       if correctness_frac == 1.0

  vs_oracle = (baseline_ms / candidate_ms) / ref_speedup   # live pair / baked calibration
            = candidate_speedup / oracle_speedup

A correct-but-slow no-op (candidate == baseline) scores ~ 0.5 + 0.5/ref_speedup.
Plausibility guard: vs_oracle > 2.0 -> hard_fail implausible_vs_oracle.
Dual-emit: also surface raw `speedup` so the stock loop16 reward_moves bar (raw >2x
swing) fires while scoring stays gated.
"""
import json
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

# plausibility guard (the oracle is a tuned production tensor-core GEMM)
if corr_frac >= 1.0 and vs_oracle > 2.0:
    hard = hard + ["implausible_vs_oracle"]

if hard:
    score = 0.0
elif corr_frac < 1.0:
    score = 0.5 * corr_frac
else:
    score = 0.5 + 0.5 * vs_oracle

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
    "reward": round(score, 6),
    "score": round(score, 6),
    "mode": mode,
    "correctness_frac": corr_frac,
    "vs_oracle": round(vs_oracle, 6),
    "speedup": round(speedup, 6),
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
