#!/usr/bin/env python3
"""compute_reward for wre-spec-accept-slot-compact (implement from an empty stub).

Reward (performance class):

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0     strictly in [0.0, 1.0]

This lane's raw measurement is vs_oracle = oracle_ms / candidate_ms, because the
start state is an untimeable empty stub. The reward formula's `speedup` is measured against a
correct-but-slow reference (here: the baseline2 variant), so

    baseline_ms = oracle_ms * ref_speedup
    speedup     = baseline_ms / candidate_ms = vs_oracle * ref_speedup

An oracle-grade candidate has vs_oracle == 1 => speedup == ref_speedup => reward 0.5;
reaching ref_speedup**2 caps at 1.0.

HARD PRE-GATES (any one hit => reward 0.0, the formula is never evaluated):
build/import failure, ANY correctness case fail, cheating, a forbidden edit path,
speedup <= 1, ref_speedup <= 1. The build/import/correctness/anti-cheat/scope gates
are enforced upstream in test.sh (which also computes the authoritative verdict);
this script reproduces the same formula for standalone/offline use and re-checks the
two numeric pre-gates itself.
"""
import json
import math
import sys

TOKEN = "WRE_ACCEPT_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit("no WRE_ACCEPT_RESULT in %s" % path)


def main():
    # argv: candidate_timing_out oracle_ms [ref_speedup]
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    ref = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cm = cand.get("timing_ms", -1)

    hard = []
    vs_oracle = (oracle_ms / cm) if (cm and cm > 0 and oracle_ms > 0) else 0.0
    if not (math.isfinite(vs_oracle) and vs_oracle > 0.0):
        hard.append("timing_invalid")
    if ref is None or not (math.isfinite(ref) and ref > 1.0):
        hard.append("ref_speedup_invalid")

    speedup = vs_oracle * ref if (ref and vs_oracle > 0) else 0.0
    base_ms = oracle_ms * ref if (ref and oracle_ms > 0) else -1.0
    if "timing_invalid" not in hard and speedup <= 1.0:
        hard.append("speedup_not_above_baseline")

    reward = 0.0 if hard else max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0))))

    print(json.dumps({
        "task_type": "performance",
        "reward": round(reward, 6),
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; 0 if any hard pre-gate hit",
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref,
        "cv": None,
        "vs_oracle": round(vs_oracle, 6),
        "oracle_ms": oracle_ms,
        "baseline_ms": round(base_ms, 6) if base_ms > 0 else -1.0,
        "candidate_ms": cm,
    }))


if __name__ == "__main__":
    main()
