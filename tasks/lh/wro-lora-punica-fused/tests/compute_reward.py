#!/usr/bin/env python3
"""wro-lora-punica-fused compute_reward — NOTE: not invoked by tests/test.sh (test.sh computes
the verdict inline); kept for documentation/consistency only.
reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; speedup = median(baseline_ms/candidate_ms)
over ABBA-paired runs, where the baseline is the frozen degraded tree materialized from the baked
baseline commit (`git show HEAD:<scope>`), never the index (see test.sh header (b)).
Hard-fails to 0.0 if ref_speedup<=1 or speedup<=NOOP_FLOOR (1.10 -- a measured noise margin: the
unedited tree really does clock 0.9896-1.023 on this image/lane, see test.sh header (d)), so a
no-op or a negative/degrading change scores exactly 0, not the old raw uncapped ratio ~1.0.
Reads the candidate + baseline timing JSON emitted by workload.py; correctness gate is enforced
in test.sh before this runs (a candidate that fails the relative-norm parity check scores 0 there)."""
import json, math, sys

NOOP_FLOOR = 1.10   # keep in sync with tests/test.sh


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith("WRO_LORA_RESULT "):
                return json.loads(line[len("WRO_LORA_RESULT "):])
    raise SystemExit(f"no WRO_LORA_RESULT in {path}")


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
    if speedup <= NOOP_FLOOR:
        hard_fail_reasons.append("speedup_not_above_1")
    elif ref_speedup <= 1.0:
        hard_fail_reasons.append("ref_speedup_not_above_1")
    else:
        reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))
    out = {
        "task_type": "performance",
        "reward": round(reward, 6),
        "hard_fail_reasons": hard_fail_reasons,
        "hard_fails": hard_fail_reasons,
        "speedup": round(speedup, 4),
        "ref_speedup": ref_speedup,
        "baseline_ms": round(base_ms, 4),
        "candidate_ms": round(cand_ms, 4),
        "metadata": {"noop_floor": NOOP_FLOOR},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
