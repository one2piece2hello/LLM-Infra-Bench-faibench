#!/usr/bin/env python3
"""compute_reward for wro-torchtitan-varlen-cu-seqlens (performance class).

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0  
  - speedup = baseline_ms / candidate_ms (wall-clock, vs the frozen degraded baseline)
  - 0.5 at parity with the oracle, capped 1.0 at oracle^2, strictly in [0, 1]
  - PRE-GATES => reward 0 without entering the formula: speedup <= 1 (never crossed
    the baseline) or ref_speedup <= 1 (invalid oracle anchor).
The correctness / scope / import-origin / anti-cheat HARD gates are enforced in
test.sh BEFORE this runs (a candidate that fails them scores 0 there); test.sh is
also the authoritative reward writer. This CLI mirrors the same formula.
"""
import json
import math
import sys

MARK = "WRO_VARLEN_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
    raise SystemExit(f"no {MARK.strip()} in {path}")


def main():
    # argv: candidate_out baseline_out [ref_speedup]
    cand = load(sys.argv[1])
    base = load(sys.argv[2])
    ref = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cm = cand.get("timing_ms", -1)
    bm = base.get("timing_ms", -1)
    speedup = (bm / cm) if (cm and cm > 0 and bm and bm > 0) else 0.0

    hard = []
    reward = 0.0
    if not (math.isfinite(speedup) and speedup > 0.0):
        hard.append("timing_invalid")
    elif speedup <= 1.0:
        hard.append("no_speedup_over_baseline")
    elif not ref or not math.isfinite(ref) or ref <= 1.0:
        hard.append("invalid_ref_speedup")
    else:
        reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))
        if not math.isfinite(reward) or reward < 0.0:
            reward, hard = 0.0, ["reward_computation_failed"]

    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref,
        "cv": {"baseline": None, "candidate": None},
        "baseline_ms": bm,
        "candidate_ms": cm,
    }
    if ref:
        out["metadata"] = {"vs_oracle_ratio": round(speedup / ref, 4), "ref_speedup": ref}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
