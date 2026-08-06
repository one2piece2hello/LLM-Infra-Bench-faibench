#!/usr/bin/env python3
"""compute_reward for s4-fft-longconv (IMPLEMENTATION class).

reward = 1.0 iff every test case passes and there is no cheat / scope violation,
reward = 0.0 otherwise   [implementation class]

The authoritative verdict is written by test.sh, which owns the correctness /
scope-diff / import-origin / anti-cheat HARD gates (the baked hollow stub raises
NotImplementedError, so the starter fails correctness and scores 0). This CLI
mirrors the same binary rule and reports the measured vs_oracle ratio as
DIAGNOSTIC METADATA only — never as the reward.

Why binary rather than the performance log formula: this task has NO timeable
baseline (the starter does not run), and its only timing anchor is the oracle
itself, so ref_speedup ~= 1.0 and min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0 is
degenerate/undefined. reward.md's legality rule for the starter package
("implementation class must fail at least one testcase") is what this satisfies.
"""
import json
import sys

TOKEN = "WRE_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no {TOKEN.strip()} in {path}")


def main():
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cm = cand.get("timing_ms", -1)

    # correctness is the AND of every case; the workload reports it as one flag.
    correctness_ok = bool(cand.get("correctness_ok", False))
    hard = [] if correctness_ok else ["correctness_failed"]
    reward = 1.0 if (correctness_ok and not hard) else 0.0

    vs_oracle = (oracle_ms / cm) if (cm and cm > 0 and oracle_ms > 0) else 0.0
    print(json.dumps({
        "task_type": "implementation",
        "reward": reward,
        "reward_formula": "implementation: 1.0 iff every case passes and no cheat, else 0.0",
        "hard_fail_reasons": hard,
        "tests": {"passed": (1 if correctness_ok else 0), "total": 1,
                  "note": "correctness_ok is the AND over every hidden case"},
        "metadata": {"vs_oracle": round(vs_oracle, 6), "oracle_ms": oracle_ms,
                     "candidate_ms": cm,
                     "note": "vs_oracle is diagnostic metadata only, never the reward"},
        "vs_oracle": round(vs_oracle, 6),
        "oracle_ms": oracle_ms,
        "candidate_ms": cm,
    }))


if __name__ == "__main__":
    main()
