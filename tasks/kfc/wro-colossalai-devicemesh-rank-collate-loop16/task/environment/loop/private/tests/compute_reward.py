#!/usr/bin/env python3
"""wro-colossalai-devicemesh-rank-collate compute_reward — reward = raw speedup (baseline_ms / candidate_ms).
Records vs_oracle_ratio = speedup / ref_speedup as metadata ONLY (never the reward). noop~1.0.
Reads the candidate + baseline timing JSON emitted by workload.py; correctness gate is enforced
in test.sh before this runs (a candidate that fails the group-equality check scores 0 there)."""
import json, sys

def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith("WRO_GDN_RESULT "):
                return json.loads(line[len("WRO_GDN_RESULT "):])
    raise SystemExit(f"no WRO_GDN_RESULT in {path}")

def main():
    # argv: candidate_out baseline_out  [ref_speedup]
    cand = load(sys.argv[1])
    base = load(sys.argv[2])
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else None
    cand_ms = cand["timing_ms"]
    base_ms = base["timing_ms"]
    speedup = base_ms / cand_ms if cand_ms > 0 else 0.0
    reward = speedup                               # RAW speedup, uncapped; no-op ~ 1.0
    out = {
        "reward": round(reward, 6),
        "baseline_ms": round(base_ms, 4),
        "candidate_ms": round(cand_ms, 4),
        "speedup": round(speedup, 4),
    }
    if ref_speedup:
        out["metadata"] = {"vs_oracle_ratio": round(speedup / ref_speedup, 4),
                           "ref_speedup": ref_speedup}   # granite-scale METADATA only, never reward
    print(json.dumps(out))

if __name__ == "__main__":
    main()
