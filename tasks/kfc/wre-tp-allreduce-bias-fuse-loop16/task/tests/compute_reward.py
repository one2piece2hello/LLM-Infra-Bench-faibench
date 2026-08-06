#!/usr/bin/env python3
"""compute_reward for wre-tp-allreduce-bias-fuse-loop16 — standalone reward helper.

reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0       

This lane measures vs_oracle = oracle_ms / candidate_ms, i.e. speedup / ref_speedup, where
ref_speedup is the ORACLE's speedup over the naive baseline. So equivalently
    reward = min(1.0, ln(vs_oracle) / ln(ref_speedup))
    matching the oracle (vs_oracle = 1) -> 0.0 (must EXCEED it)
    vs_oracle = ref_speedup (i.e. speedup = ref_speedup^2) -> 1.0 cap
    at or below the oracle (speedup <= ref_speedup) -> 0.0
    value range strictly [0.0, 1.0]

HARD pre-gates (any one hit => reward 0, formula NOT entered): the correctness gate in test.sh
(a candidate failing the reference match scores 0 there), speedup <= 1, ref_speedup <= 1,
bad timings. oracle_ms is a calibrated held-out constant.

usage: compute_reward.py <candidate_timing_out> <oracle_ms> [ref_speedup]
NOTE: the authoritative verdict is the one test.sh writes to /logs/verifier/reward.json; this
helper exists so the lane's reward definition is reproducible standalone.
"""
import json, math, os, sys

TOKEN = "WRE_RESULT "


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
    try:
        return float(open(os.path.join(here, "ref_speedup.txt")).read().strip())
    except Exception:
        pass
    try:
        v = float(json.load(open("/opt/verifier-correctness-manifest.json")).get("ref_speedup", 0) or 0)
        return v
    except Exception:
        return 0.0


def log_reward(speedup, ref):
    """min(1.0, ln(speedup/ref)/ln(ref)) if speedup > ref else 0.0 clamped to [0,1]; 0 unless speedup>1 and ref>1."""
    if not (math.isfinite(speedup) and math.isfinite(ref)) or speedup <= 1.0 or ref <= 1.0:
        return 0.0
    return max(0.0, min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0))))


def main():
    cand = load(sys.argv[1])
    oracle_ms = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    cand_ms = float(cand.get("timing_ms", -1))
    ref = read_ref(sys.argv)
    vs_oracle = (oracle_ms / cand_ms) if (oracle_ms > 0 and cand_ms > 0) else 0.0
    speedup = vs_oracle * ref if (vs_oracle > 0 and ref > 0) else 0.0

    hard = []
    if not (oracle_ms > 0 and cand_ms > 0):
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
        "vs_oracle": round(vs_oracle, 6),
        "candidate_ms": round(cand_ms, 4),
        "oracle_ms": round(oracle_ms, 4) if oracle_ms > 0 else None,
    }))


if __name__ == "__main__":
    main()
