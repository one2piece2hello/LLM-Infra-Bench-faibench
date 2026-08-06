#!/usr/bin/env python3
"""compute_reward for wre-kvoffload-lfu-admit-sketch-loop16 — PERFORMANCE reward formula.

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, value range [0.0, 1.0]
  * speedup == ref_speedup -> 0.00
  * speedup >= ref_speedup ** 2 -> 1.00 (capped)
Pre-gates enforced HERE and, authoritatively, in tests/test.sh before this runs
(scope / import-origin / trusted-restore / anti-cheat / correctness): any hard
fail, a correctness FAIL, speedup <= 1 or ref_speedup <= 1 all yield reward 0
WITHOUT entering the formula.

NOTE ON WIRING: tests/test.sh of this task computes the verdict INLINE and does
not exec this module; it is kept (and kept correct) because the trusted-restore
gate requires the file to be present, and because external harnesses import it.
"""
import json
import math
import sys

TOKEN = "WRE_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(TOKEN):
                return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRE_RESULTin {{path}}")


def log_reward(speedup, ref_speedup):
    """The performance reward formula + its pre-gates. Returns (reward, hard_fails)."""
    hard = []
    if not (isinstance(speedup, (int, float)) and math.isfinite(speedup) and speedup > 1.0):
        hard.append("speedup_not_above_baseline")
    if not (isinstance(ref_speedup, (int, float)) and math.isfinite(ref_speedup) and ref_speedup > 1.0):
        hard.append("ref_speedup_invalid")
    if hard:
        return 0.0, hard
    return round(min(1.0, max(0.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))), 6), []


def main():
    # argv: candidate_timing_out oracle_ms [ref_speedup]
    # B1-empty has no timeable baseline: the baseline is the one implied by the
    # anchor, baseline_ms = oracle_ms * ref_speedup, so speedup is candidate-vs-
    # baseline exactly as the reward formula requires. vs_oracle is kept as metadata.
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ref = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cand_ms = cand.get("timing_ms", -1)
    vs_oracle = (oracle_ms / cand_ms) if (oracle_ms and cand_ms and cand_ms > 0) else 0.0
    base_ms = oracle_ms * ref if oracle_ms > 0 else -1.0
    speedup = (base_ms / cand_ms) if (cand_ms and cand_ms > 0 and base_ms > 0) else 0.0
    reward, hard = log_reward(speedup, ref)
    print(json.dumps({
        "schema_version": "kernelbench_reward_v3_oracle_relative",
        "task_type": "performance",
        "reward": reward,
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref,
        "cv": {"baseline": None, "candidate": None},
        "vs_oracle": round(vs_oracle, 6),
        "oracle_ms": round(oracle_ms, 4) if oracle_ms else None,
        "baseline_ms": round(base_ms, 4) if base_ms > 0 else None,
        "candidate_ms": round(cand_ms, 4),
    }))


if __name__ == "__main__":
    main()
