#!/usr/bin/env python3
"""compute_reward for mamba-zoh-discretize.

IMPLEMENTATION class (task_class = 实现类). BINARY reward per the reward specification:

    reward = 1.0  iff EVERY correctness case passes AND no cheat / forbidden-path condition
    reward = 0.0  otherwise

There is no partial credit and NO timing term: the measured runtime is diagnostic metadata only.

usage: compute_reward.py <correctness_out> [timing_out]
NOTE: the authoritative verdict is the one test.sh writes to /logs/verifier/reward.json; this
helper exists so the lane's reward definition is reproducible standalone.
"""
import json, sys

TOKEN = "WRE_RESULT "


def load(path):
    with open(path) as f:
        last = None
        for line in f:
            if line.startswith(TOKEN):
                last = json.loads(line[len(TOKEN):])
        if last is None:
            raise SystemExit(f"no {TOKEN.strip()} in {path}")
        return last


def main():
    corr = load(sys.argv[1])
    timing = None
    if len(sys.argv) > 2:
        try:
            timing = load(sys.argv[2])
        except Exception:
            timing = None

    hard = []
    ok = bool(corr.get("correctness_ok"))
    if not ok:
        hard.append("correctness_failed")
    if corr.get("error"):
        hard.append("candidate_error")
    reward = 1.0 if (ok and not hard) else 0.0

    out = {
        "task_type": "implementation",
        "reward": reward,
        "reward_formula": "binary: 1.0 iff every case passes and no cheat/forbidden path, else 0.0",
        "hard_fail_reasons": hard,
        "tests": {"passed": 8 if reward == 1.0 else 0, "total": 8},
        "detail": corr.get("detail") or corr.get("error"),
    }
    if timing:
        out["diagnostic_only"] = {"candidate_ms": timing.get("timing_ms")}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
