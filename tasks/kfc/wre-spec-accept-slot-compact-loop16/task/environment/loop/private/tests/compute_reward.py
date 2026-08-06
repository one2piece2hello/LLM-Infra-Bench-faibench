#!/usr/bin/env python3
"""compute_reward for wre-spec-accept-slot-compact (implement from an empty stub).
reward = oracle_ms / candidate_ms (vs the calibrated oracle anchor), 0 when the
correctness gate in test.sh fails (the baked hollow stub raises)."""
import json
import sys

TOKEN = "WRE_ACCEPT_RESULT "


def load(path):
    for line in open(path):
        if line.startswith(TOKEN):
            return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no WRE_ACCEPT_RESULT in {path}")


def main():
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cm = cand.get("timing_ms", -1)
    r = (oracle_ms / cm) if (cm and cm > 0 and oracle_ms > 0) else 0.0
    print(json.dumps({"reward": round(r, 6), "vs_oracle": round(r, 6),
                      "oracle_ms": oracle_ms, "candidate_ms": cm}))


if __name__ == "__main__":
    main()
