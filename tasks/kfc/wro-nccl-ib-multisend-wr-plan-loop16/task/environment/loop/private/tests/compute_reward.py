#!/usr/bin/env python3
"""compute_reward for wro-nccl-ib-multisend-wr-plan (acceleration).
reward = base_ms / candidate_ms (raw wall speedup vs the frozen degraded baseline),
gated by correctness in test.sh (wrong output -> reward 0 there)."""
import json
import sys

TOKEN = "WRO_IBMS_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRO_IBMS_RESULT in {path}")


def main():
    cand = load(sys.argv[1])
    base_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cm = cand.get("timing_ms", -1)
    speedup = (base_ms / cm) if (cm and cm > 0 and base_ms > 0) else 0.0
    print(json.dumps({"reward": round(speedup, 6), "speedup": round(speedup, 6),
                      "baseline_ms": base_ms, "candidate_ms": cm}))


if __name__ == "__main__":
    main()
