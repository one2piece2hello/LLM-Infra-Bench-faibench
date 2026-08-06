#!/usr/bin/env python3
"""wro-ssm-ssd-chunkscan compute_reward — NOTE: not invoked by tests/test.sh (test.sh computes
the verdict inline); kept for documentation/consistency only.
reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; speedup = median(baseline_ms/candidate_ms)
over ABBA-paired runs. Hard-fails to 0.0 if speedup<=1 or ref_speedup<=1 (so a no-op or a
negative/degrading change scores exactly 0, not the old raw uncapped ratio ~1.0).
Reads the candidate + baseline timing JSON emitted by workload.py; correctness gate is enforced
in test.sh before this runs (a candidate that fails the decision-trace match scores 0 there)."""
import json, math, sys

def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith("WRO_SSM_RESULT "):
                return json.loads(line[len("WRO_SSM_RESULT "):])
    raise SystemExit(f"no WRO_SSM_RESULT in {path}")

def main():
    # argv: candidate_out baseline_out  [ref_speedup]
    cand = load(sys.argv[1])
    base = load(sys.argv[2])
    ref_speedup = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cand_ms = cand["timing_ms"]
    base_ms = base["timing_ms"]
    speedup = base_ms / cand_ms if cand_ms > 0 else 0.0
    hard_fail_reasons = []
    reward = 0.0
    if speedup <= 1.0:
        hard_fail_reasons.append("speedup_not_above_1")
    elif ref_speedup <= 1.0:
        hard_fail_reasons.append("ref_speedup_not_above_1")
    else:
        reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))
    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": hard_fail_reasons,
        "speedup": round(speedup, 4),
        "ref_speedup": ref_speedup,
        "baseline_ms": round(base_ms, 4),
        "candidate_ms": round(cand_ms, 4),
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
