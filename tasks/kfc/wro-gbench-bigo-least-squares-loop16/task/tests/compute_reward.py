#!/usr/bin/env python3
"""compute_reward for wro-gbench-bigo-least-squares-loop16 — PERFORMANCE reward formula.

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0   in [0.0, 1.0]
  speedup     = base_ms / candidate_ms (raw wall speedup vs the frozen degraded baseline)
  ref_speedup = the oracle's median speedup in the same image (unchanged source)
  parity with the oracle -> 0.5 ; oracle^2 or better -> capped 1.0

PRE-GATES (any hit -> reward 0, the formula is NOT entered): the correctness /
scope / import-origin / anti-cheat gates live in test.sh and score 0 there;
here the remaining pre-gates are speedup <= 1 and ref_speedup <= 1.

NOTE: test.sh computes the authoritative reward inline with the identical
formula. This module is the standalone/argv entry point kept for parity.
"""
import json
import math
import sys

TOKEN = "WRO_BIGO_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRO_BIGO_RESULT in {path}")


def log_reward(speedup, ref_speedup):
    """The performance reward curve + its numeric pre-gates. Returns (reward, reasons)."""
    reasons = []
    if not (isinstance(speedup, (int, float)) and math.isfinite(speedup)):
        return 0.0, ["invalid_primary_metric_value"]
    if not (isinstance(ref_speedup, (int, float)) and math.isfinite(ref_speedup)):
        return 0.0, ["invalid_ref_speedup"]
    if speedup <= 1.0:
        return 0.0, ["speedup_not_above_baseline"]
    if ref_speedup <= 1.0:
        return 0.0, ["ref_speedup_invalid"]
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))), reasons


def main():
    # argv: candidate_timing_out baseline_ms [ref_speedup]
    cand = load(sys.argv[1])
    base_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cm = cand.get("timing_ms", -1)
    speedup = (base_ms / cm) if (cm and cm > 0 and base_ms > 0) else 0.0
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    reward, reasons = log_reward(speedup, ref_speedup)
    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": reasons,
        "speedup": round(speedup, 6),
        "ref_speedup": ref_speedup,
        "cv": {"baseline": None, "candidate": None},
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "baseline_ms": base_ms,
        "candidate_ms": cm,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
