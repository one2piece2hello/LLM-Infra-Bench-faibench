#!/usr/bin/env python3
"""compute_reward for wro-deepspeed-curriculum-cluster-select.
reward = baseline_ms / candidate_ms (wall-clock speedup; noop ~ 1.0), gated by
correctness in test.sh (wrong selected sample-ids -> reward 0 there). Records
vs_oracle_ratio as metadata only."""
import json
import sys


def load(path):
    for line in open(path):
        if line.startswith("WRO_CURRIC_RESULT "):
            return json.loads(line[len("WRO_CURRIC_RESULT "):])
    raise SystemExit(f"no WRO_CURRIC_RESULT in {path}")


def main():
    cand = load(sys.argv[1]); base = load(sys.argv[2])
    ref = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cm = cand.get("timing_ms", -1); bm = base.get("timing_ms", -1)
    speedup = (bm / cm) if (cm and cm > 0 and bm > 0) else 0.0
    out = {"reward": round(speedup, 6), "baseline_ms": bm, "candidate_ms": cm,
           "speedup": round(speedup, 6)}
    if ref:
        out["metadata"] = {"vs_oracle_ratio": round(speedup / ref, 4), "ref_speedup": ref}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
