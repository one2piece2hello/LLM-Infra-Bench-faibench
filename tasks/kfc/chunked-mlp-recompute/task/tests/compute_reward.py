#!/usr/bin/env python3
"""compute_reward for chunked-mlp-recompute, task_class = 实现类.

Reward (IMPLEMENTATION class):

    reward = 1.0   iff EVERY visible correctness case passes AND no cheating
    reward = 0.0   otherwise (any case fail, or any anti-cheat / forbidden-path hit)

BINARY, never a ratio. The previous scoring used the peak-memory ratio
vs_oracle = oracle_peak_bytes / candidate_peak_bytes as the reward; that is now a
REPORTED DIAGNOSTIC only (peak_bytes / vs_oracle), because reward.md forbids partial
credit for an implementation task. Peak memory remains the thing that makes the task
hard, and the correctness gate in test.sh (fp32 reference on y AND dx, plus the CSPRNG
anti-cache probe) is what decides the score.

Reads the candidate timing JSON emitted by workload.py (peak GPU bytes in the
`timing_ms` field). test.sh computes the authoritative verdict; this script reproduces
the same rule for standalone/offline use.
"""
import json
import sys


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith("WRE_RESULT "):
                return json.loads(line[len("WRE_RESULT "):])
    raise SystemExit(f"no WRE_RESULT in {path}")


def main():
    # argv: candidate_timing_out [oracle_peak_bytes] [correctness_out]
    cand = load(sys.argv[1])
    oracle_bytes = float(sys.argv[2]) if len(sys.argv) > 2 else None
    cand_bytes = cand.get("timing_ms", -1)          # peak GPU bytes for this task

    hard = []
    passed = total = 0
    # Correctness is the ONLY thing that decides the reward. When the correctness
    # output is supplied, read the case counts from it; otherwise report unknown and
    # fail closed (test.sh is the authoritative path and always supplies the gate).
    if len(sys.argv) > 3:
        corr = load(sys.argv[3])
        passed = int(corr.get("cases_passed", 0) or 0)
        total = int(corr.get("cases_total", 0) or 0)
        if not corr.get("correctness_ok"):
            hard.append("correctness_failed")
        elif total > 0 and passed < total:
            hard.append("correctness_cases_incomplete")
    else:
        hard.append("correctness_output_not_supplied")

    if not (cand_bytes and cand_bytes > 0):
        hard.append("timing_invalid")

    reward = 0.0 if hard else 1.0
    vs_oracle = (oracle_bytes / cand_bytes) if (oracle_bytes and cand_bytes and cand_bytes > 0) else 0.0

    print(json.dumps({
        "task_type": "implementation",
        "reward": round(reward, 6),
        "reward_formula": "binary: 1.0 iff all cases pass and no cheating, else 0.0",
        "hard_fail_reasons": hard,
        "tests": {"passed": passed, "total": total},
        # diagnostics — reported, never scaled into the reward
        "peak_bytes": round(cand_bytes, 4) if cand_bytes else -1.0,
        "oracle_peak_bytes": round(oracle_bytes, 4) if oracle_bytes else None,
        "vs_oracle": round(vs_oracle, 6),
        "speedup": round(vs_oracle, 6),   # loop harness reads dev_speedup off this key (§X)
    }))


if __name__ == "__main__":
    main()
