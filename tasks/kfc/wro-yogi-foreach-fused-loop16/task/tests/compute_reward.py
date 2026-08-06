#!/usr/bin/env python3
"""wro-yogi-foreach-fused-loop16 compute_reward — PERFORMANCE reward formula.

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0          range [0.0, 1.0]

speedup = baseline_ms / candidate_ms (vs the frozen degraded baseline);
ref_speedup = the oracle's speedup in the same image. Matching the oracle scores
0.5, reaching oracle^2 caps at 1.0.

Pre-gates (any one hit => reward 0.0, the formula is not entered): the correctness
gate in test.sh (which owns scope_ok / import_origin_ok / correctness_ok and writes
the authoritative verdict), speedup <= 1, ref_speedup <= 1.

NOTE: test.sh computes the authoritative reward inline and only presence-checks this
file. This module is the same formula, kept for standalone/manual use.
"""
import json, math, sys

TOKEN = "WRO_GDN_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(TOKEN):
                return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRO_GDN_RESULT in {path}")


def main():
    # argv: candidate_out baseline_out_or_baseline_ms [ref_speedup]
    cand = load(sys.argv[1])
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        base_ms = float(arg2)
    except (TypeError, ValueError):
        base_ms = load(arg2)["timing_ms"]
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cand_ms = cand.get("timing_ms", -1)
    speedup = (base_ms / cand_ms) if (cand_ms and cand_ms > 0 and base_ms > 0) else 0.0
    hard = []
    if not (math.isfinite(speedup) and math.isfinite(ref_speedup)):
        hard.append("non_finite_metric")
    elif ref_speedup <= 1.0:
        hard.append("ref_speedup_invalid")
    elif speedup <= 1.0:
        hard.append("speedup_not_above_baseline")
    reward = 0.0
    if not hard:
        reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))
        reward = max(0.0, min(1.0, reward))
    print(json.dumps({
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref_speedup,
        "cv": {"baseline": None, "candidate": None},
        "baseline_ms": round(base_ms, 4),
        "candidate_ms": round(cand_ms, 4),
    }))


if __name__ == "__main__":
    main()
