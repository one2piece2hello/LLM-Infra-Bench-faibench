#!/usr/bin/env python3
"""wre-runtime-memplan-arena-loop16 compute_reward. reward = vs_oracle = oracle_ms/candidate_ms.
The correctness gate is enforced in test.sh (a candidate that fails the fp32-reference match scores
0 there). oracle_ms is a calibrated held-out constant; the empty/stub start cannot be timed, so the
1.0 anchor is the ORACLE (matching oracle=1.0, eager baseline2 in (0,1), faster>1). Reads the
candidate + timing JSON emitted by workload.py."""
import json, sys


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith("WRE_RESULT "):
                return json.loads(line[len("WRE_RESULT "):])
    raise SystemExit(f"no WRE_RESULT in {path}")


def main():
    # argv: candidate_timing_out oracle_ms
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else None
    cand_ms = cand.get("timing_ms", -1)
    vs_oracle = (oracle_ms / cand_ms) if (oracle_ms and cand_ms and cand_ms > 0) else 0.0
    out = {
        "reward": round(vs_oracle, 6),          # == vs_oracle; oracle=1.0, eager<1, faster>1
        "speedup": round(vs_oracle, 6),         # loop harness reads dev_speedup off this key (§X)
        "vs_oracle": round(vs_oracle, 6),
        "candidate_ms": round(cand_ms, 4),
        "oracle_ms": round(oracle_ms, 4) if oracle_ms else None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
