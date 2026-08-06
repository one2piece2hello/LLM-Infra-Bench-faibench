#!/usr/bin/env python3
"""Canonical kernelbench reward writer — the reward specification.

Task-agnostic; do NOT special-case a task here.

Dual-mode:
  - implementation task (verifier_state.task_kind == "correctness"):
        BINARY. reward = 1.0 iff every hard gate is true, no hard_fail_reasons and
        every test case passed; else 0.0. A single failing case scores 0.
  - performance task (acceleration):
        reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
        (matching the oracle -> 0.5; reaching oracle^2 -> capped 1.0); range [0, 1].
        Pre-gates -> reward = 0 WITHOUT entering the formula: build/import failure,
        any correctness case fail, anti-cheat, forbidden edit path, speedup <= 1,
        ref_speedup <= 1.

Reads:  /logs/verifier/verifier_state.json, /logs/verifier/benchmark_results.json
Writes: /logs/verifier/reward.json, /logs/verifier/reward.txt
"""
import json, math
from pathlib import Path

LOG = Path("/logs/verifier")
STATE = LOG / "verifier_state.json"
BENCH = LOG / "benchmark_results.json"
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


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE, {})
    bench = read_json(BENCH, {})
    hard = list(state.get("hard_fail_reasons") or [])
    task_kind = state.get("task_kind") or bench.get("task_kind") or "acceleration"
    gates_ok = all(bool(state.get(g)) for g in GATES)

    # case accounting — binary correctness: EVERY case must pass
    def _int(v):
        try:
            return int(v)
        except Exception:
            return 0

    exp_cases = _int(state.get("expected_case_count"))
    act_cases = _int(state.get("actual_case_count"))
    cases_ok = (exp_cases <= 0) or (act_cases >= exp_cases)

    speedup = ref = 1.0
    reward = 0.0
    if task_kind == "correctness":
        # ---- implementation task: BINARY (reward.md) ----
        reward = 1.0 if (gates_ok and not hard and cases_ok) else 0.0
        if reward == 0.0 and not hard:
            hard.append("correctness_or_validation_incomplete")
    else:
        # ---- performance task: log formula anchored on ref_speedup (reward.md) ----
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            hard.append("invalid_primary_metric_value")

        # pre-gates: any hit => reward 0, the formula is NOT entered
        if not (gates_ok and cases_ok):
            if not hard:
                hard.append("validation_incomplete")
        elif not (math.isfinite(speedup) and math.isfinite(ref)):
            hard.append("invalid_primary_metric_value")
        elif speedup <= 1.0:
            hard.append("no_speedup_over_baseline")
        elif ref <= 1.0:
            hard.append("invalid_ref_speedup")

        if not hard:
            reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))
            if not math.isfinite(reward) or reward < 0.0:
                reward = 0.0
                hard.append("reward_computation_failed")

    cv = bench.get("cv")
    if not isinstance(cv, dict):
        cv = {"baseline": bench.get("baseline_cv"), "candidate": bench.get("candidate_cv")}

    out = {
        "schema_version": "kernelbench_reward_v2_logratio",
        "task_kind": task_kind,
        "task_type": ("implementation" if task_kind == "correctness" else "performance"),
        "reward": reward,
        "speedup": speedup,
        "ref_speedup": ref,
        "cv": cv,
        "reward_formula": ("implementation: 1.0 iff every case passes and no cheat, else 0.0"
                           if task_kind == "correctness"
                           else "performance: min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0"),
        "tests": {"passed": act_cases, "total": exp_cases},
        "actual_hardware_type": state.get("actual_hardware_type", "") or bench.get("actual_hardware_type", ""),
        "hard_fail_reasons": hard,
        "state_path": str(STATE),
        "benchmark_path": str(BENCH),
    }
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
