#!/usr/bin/env python3
"""compute_reward for wro-gbench-counter-finalize-loop16 — standalone reward helper.

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0       
  speedup = baseline_ms / candidate_ms  (absolute, vs the frozen degraded baseline)
  matching the oracle -> 0.5 ; oracle^2 -> 1.0 cap ; at/below the baseline -> 0.0
  value range strictly [0.0, 1.0]

HARD pre-gates (any one hit => reward 0, formula NOT entered): the correctness gate in test.sh
(a candidate that fails parity scores 0 there), speedup <= 1, ref_speedup <= 1, bad timings.

usage: compute_reward.py <candidate_timing_out> <baseline_ms|baseline_timing_out> [ref_speedup]
NOTE: the authoritative verdict is the one test.sh writes to /logs/verifier/reward.json; this
helper exists so the lane's reward definition is reproducible standalone.
"""
import json, math, os, sys

TOKEN = "WRO_CNT_RESULT "


def load(path):
    with open(path) as f:
        for line in f:
            if line.startswith(TOKEN):
                return json.loads(line[len(TOKEN):])
    raise SystemExit(f"no {TOKEN.strip()} in {path}")


def read_ref(argv):
    if len(argv) > 3:
        try:
            return float(argv[3])
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("/opt/verifier-correctness-manifest.json",):
        try:
            v = float(json.load(open(p)).get("ref_speedup", 0) or 0)
            if v > 1.0:
                return v
        except Exception:
            pass
    try:
        return float(open(os.path.join(here, "ref_speedup.txt")).read().strip())
    except Exception:
        return 0.0


def log_reward(speedup, ref):
    """min(1.0, ln(speedup/ref)/ln(ref)) if speedup > ref else 0.0 clamped to [0,1]; 0 unless speedup>1 and ref>1."""
    if not (math.isfinite(speedup) and math.isfinite(ref)) or speedup <= 1.0 or ref <= 1.0:
        return 0.0
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0))))


def main():
    cand = load(sys.argv[1])
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "-1"
    try:
        base_ms = float(arg2)
    except ValueError:
        base_ms = float(load(arg2).get("timing_ms", -1))
    cand_ms = float(cand.get("timing_ms", -1))
    ref = read_ref(sys.argv)
    speedup = (base_ms / cand_ms) if (cand_ms > 0 and base_ms > 0) else 0.0

    hard = []
    if not (cand_ms > 0 and base_ms > 0):
        hard.append("timing_invalid")
    if not math.isfinite(ref) or ref <= 1.0:
        hard.append("ref_speedup_invalid")
    if not math.isfinite(speedup) or speedup <= 1.0:
        hard.append("speedup_not_above_baseline")
    reward = 0.0 if hard else log_reward(speedup, ref)

    print(json.dumps({
        "task_type": "performance",
        "reward": round(reward, 6),
        "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
        "hard_fail_reasons": hard,
        "speedup": round(speedup, 6),
        "ref_speedup": ref,
        "cv": {},
        "baseline_ms": round(base_ms, 4),
        "candidate_ms": round(cand_ms, 4),
    }))


if __name__ == "__main__":
    main()
