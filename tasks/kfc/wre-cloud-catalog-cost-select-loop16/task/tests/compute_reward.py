#!/usr/bin/env python3
"""compute_reward for wre-cloud-catalog-cost-select-loop16.

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0   in [0.0, 1.0]

This lane's harness measures vs_oracle = oracle_ms / candidate_ms directly (the
empty/stub start cannot be timed, so the anchor is the ORACLE). The absolute
speedup the reward.md curve consumes is

    speedup = vs_oracle * ref_speedup

where ref_speedup = oracle_ms / baseline_ms is the oracle's speedup over the
naive/degraded reference in the SAME image (source: tests/ref_speedup.txt,
uploaded fresh; unchanged numeric provenance). Parity with the oracle -> 0.5;
oracle^2 or better -> capped 1.0.

PRE-GATES (any hit -> reward 0, formula NOT entered): the correctness / scope /
import-origin / anti-cheat gates live in test.sh and score 0 there; here the
remaining numeric pre-gates are speedup <= 1 and ref_speedup <= 1.

NOTE: test.sh computes the authoritative reward inline with the identical
formula. This module is the standalone/argv entry point kept for parity.
"""
import json
import math
import sys

TOKEN = "WRE_CAT_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(TOKEN):
                return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRE_CAT_RESULT in {path}")


def log_reward(speedup, ref_speedup):
    """reward.md performance curve + its numeric pre-gates. Returns (reward, reasons)."""
    if not (isinstance(speedup, (int, float)) and math.isfinite(speedup)):
        return 0.0, ["invalid_primary_metric_value"]
    if not (isinstance(ref_speedup, (int, float)) and math.isfinite(ref_speedup)):
        return 0.0, ["invalid_ref_speedup"]
    if speedup <= 1.0:
        return 0.0, ["speedup_not_above_baseline"]
    if ref_speedup <= 1.0:
        return 0.0, ["ref_speedup_invalid"]
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))), []


def main():
    # argv: candidate_timing_out oracle_ms [ref_speedup]
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cand_ms = cand.get("timing_ms", -1)
    vs_oracle = (oracle_ms / cand_ms) if (oracle_ms and oracle_ms > 0 and cand_ms and cand_ms > 0) else 0.0
    speedup = vs_oracle * ref_speedup
    reward, reasons = log_reward(speedup, ref_speedup)
    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": reasons,
        "speedup": round(speedup, 6),
        "ref_speedup": ref_speedup,
        "cv": {"baseline": None, "candidate": None},
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "vs_oracle": round(vs_oracle, 6),
        "candidate_ms": round(cand_ms, 6) if isinstance(cand_ms, (int, float)) else cand_ms,
        "oracle_ms": round(oracle_ms, 6) if oracle_ms and oracle_ms > 0 else None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
