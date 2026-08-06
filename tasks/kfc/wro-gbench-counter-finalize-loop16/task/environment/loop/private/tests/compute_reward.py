#!/usr/bin/env python3
"""compute_reward for wro-gbench-counter-finalize (acceleration).
reward = base_ms / candidate_ms (raw wall speedup vs the frozen degraded baseline),
gated by correctness in test.sh (wrong finalized values -> reward 0 there).
noop(degraded)~=1.0; oracle>>1 (vectorized); baseline2 in between; negative=0."""
import json
import sys


def load(path):
    for line in open(path):
        if line.startswith("WRO_CNT_RESULT "):
            return json.loads(line[len("WRO_CNT_RESULT "):])
    raise SystemExit(f"no WRO_CNT_RESULT in {path}")


def main():
    cand = load(sys.argv[1])
    base_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cm = cand.get("timing_ms", -1)
    speedup = (base_ms / cm) if (cm and cm > 0 and base_ms > 0) else 0.0
    print(json.dumps({"reward": round(speedup, 6), "speedup": round(speedup, 6),
                      "baseline_ms": base_ms, "candidate_ms": cm}))


if __name__ == "__main__":
    main()
