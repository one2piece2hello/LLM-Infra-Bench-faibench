#!/usr/bin/env python3
"""Canonical kernelbench reward writer. Task-agnostic; do NOT special-case a task here.

Reward definition (the reward specification):

  performance / acceleration task:
        reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
        - speedup == ref_speedup -> 0.0 (must EXCEED it to score)   (matching the oracle)
        - speedup >= ref_speedup**2  -> 1.0   (capped)
        - value range strictly [0.0, 1.0]
        HARD pre-gates (any one hit => reward = 0, formula NOT entered):
          build/import failure, ANY correctness case fail, cheating,
          forbidden_edit_paths touched, speedup <= 1, ref_speedup <= 1.

  implementation / correctness task:
        reward = 1.0 iff EVERY visible case passes and no cheat / forbidden-path
                 condition fired; 0.0 otherwise.  BINARY — never a pass-fraction.

Reads:  /logs/verifier/verifier_state.json, /logs/verifier/benchmark_results.json,
        /logs/verifier/correctness_results.json
Writes: /logs/verifier/reward.json, /logs/verifier/reward.txt
"""
import json, math
from pathlib import Path

LOG = Path("/logs/verifier")
STATE = LOG / "verifier_state.json"
BENCH = LOG / "benchmark_results.json"
CORR = LOG / "correctness_results.json"
REWARD = LOG / "reward.json"
TEXT = LOG / "reward.txt"

GATES = [
    "correctness_ok",
    "trusted_restore_ok",
    "hidden_correctness_ok",
    "baseline_ok",
    "benchmark_ok",
    "anti_cheat_ok",
]


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def log_reward(speedup, ref):
    """reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0, clamped to [0.0, 1.0]."""
    try:
        num = math.log(float(speedup))
        den = math.log(float(ref))
    except Exception:
        return 0.0
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0.0:
        return 0.0
    r = max(0.0, min(1.0, num / den - 1.0))
    if not math.isfinite(r):
        return 0.0
    return max(0.0, min(1.0, r))


def case_counts(state, corr):
    total = state.get("expected_case_count")
    passed = state.get("actual_case_count")
    suites = corr.get("suites") or []
    if suites:
        try:
            p = sum(int(s.get("passed") or 0) for s in suites)
            t = sum(int(s.get("expected_case_count") or 0) for s in suites)
            if t > 0:
                passed, total = p, t
        except Exception:
            pass
    return passed, total


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE, {})
    bench = read_json(BENCH, {})
    corr = read_json(CORR, {})
    hard = list(state.get("hard_fail_reasons") or [])
    task_kind = state.get("task_kind") or bench.get("task_kind") or "acceleration"
    gates_ok = all(bool(state.get(g)) for g in GATES)
    passed, total = case_counts(state, corr)

    speedup = ref = 1.0
    reward = 0.0
    if task_kind == "correctness":
        # IMPLEMENTATION task -> BINARY. Any single case short of the expected
        # count zeroes the reward (no partial / graded fraction credit).
        cases_ok = True
        if total is not None and passed is not None:
            try:
                cases_ok = int(passed) >= int(total)
            except Exception:
                cases_ok = False
        if not cases_ok and "case_count_short" not in hard:
            hard.append("case_count_short")
        reward = 1.0 if (gates_ok and cases_ok and not hard) else 0.0
        if reward == 0.0 and not hard:
            hard.append("correctness_or_validation_incomplete")
    else:
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            if "invalid_primary_metric_value" not in hard:
                hard.append("invalid_primary_metric_value")
        # Correctness / anti-cheat / forbidden-path pre-gates arrive via gates_ok +
        # hard (unchanged, never weakened). The two metric pre-gates are explicit.
        if not math.isfinite(speedup) or speedup <= 1.0:
            if "speedup_not_above_baseline" not in hard:
                hard.append("speedup_not_above_baseline")
        if not math.isfinite(ref) or ref <= 1.0:
            if "ref_speedup_invalid" not in hard:
                hard.append("ref_speedup_invalid")
        if gates_ok and not hard:
            reward = log_reward(speedup, ref)

    out = {
        "schema_version": "kernelbench_reward_v2",
        "task_kind": task_kind,
        "task_type": "implementation" if task_kind == "correctness" else "performance",
        "reward": reward,
        "reward_formula": ("binary: 1.0 iff every visible case passes and no cheat/forbidden path"
                           if task_kind == "correctness"
                           else "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0"),
        "speedup": speedup,
        "ref_speedup": ref,
        "cv": bench.get("cv") or {},
        "tests": {"passed": passed, "total": total},
        "actual_hardware_type": state.get("actual_hardware_type", "") or bench.get("actual_hardware_type", ""),
        "hard_fail_reasons": hard,
        "state_path": str(STATE),
        "benchmark_path": str(BENCH),
    }
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
