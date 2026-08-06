#!/usr/bin/env python3
"""compute_reward for wro-deepspeed-curriculum-cluster-select.

Reward (performance class):

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0     strictly in [0.0, 1.0]

with speedup = baseline_ms / candidate_ms. Matching the oracle (speedup ==
ref_speedup) scores 0.5; reaching ref_speedup**2 caps at 1.0.

HARD PRE-GATES (any one hit => reward 0.0, the formula is never evaluated):
build/import failure, ANY correctness case fail, cheating, a forbidden edit path,
speedup <= 1, ref_speedup <= 1. The build/import/correctness/anti-cheat/scope
gates are enforced upstream in test.sh (which also computes the authoritative
verdict); this script reproduces the same formula for standalone/offline use and
re-checks the two numeric pre-gates itself.
"""
import json
import math
import sys

TOKEN = "WRO_CURRIC_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRO_CURRIC_RESULT in {path}")


def main():
    # argv: candidate_out baseline_out [ref_speedup]
    cand = load(sys.argv[1])
    base = load(sys.argv[2])
    ref = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cm = cand.get("timing_ms", -1)
    bm = base.get("timing_ms", -1)
    hard = []
    speedup = (bm / cm) if (cm and cm > 0 and bm and bm > 0) else 0.0
    if not (math.isfinite(speedup) and speedup > 0.0):
        hard.append("timing_invalid")
    elif speedup <= 1.0:
        hard.append("speedup_not_above_baseline")
    if ref is None or not (math.isfinite(ref) and ref > 1.0):
        hard.append("ref_speedup_invalid")

    if hard:
        reward = 0.0
    else:
        reward = max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0))))

    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; 0 if any hard pre-gate hit",
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref,
        "cv": None,
        "baseline_ms": bm,
        "candidate_ms": cm,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
