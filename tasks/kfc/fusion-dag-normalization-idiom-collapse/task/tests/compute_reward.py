#!/usr/bin/env python3
"""Canonical kernelbench reward writer. Task-agnostic; do NOT special-case a task here.

Reward formula:

  performance (acceleration):
      reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
    - speedup      = candidate speedup vs the frozen baseline (measured by test.sh)
    - ref_speedup  = oracle's median speedup in the same image (tests/ref_speedup.txt
                     or the in-image manifest; value NEVER changed here)
    - matching the oracle -> 0.5; reaching oracle^2 -> capped at 1.0
    - strict range [0.0, 1.0]
    Pre-gates (any one hit => reward 0.0, formula not entered):
      build/import failure, ANY correctness case fail, cheating, forbidden-path edit,
      speedup <= 1, ref_speedup <= 1.

  implementation (correctness):
      reward = 1.0 iff ALL cases pass and no cheating; else 0.0 (binary, never a fraction).

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

    exp_cases = state.get("expected_case_count")
    act_cases = state.get("actual_case_count")

    speedup = ref = 1.0
    reward = 0.0
    if task_kind == "correctness":
        # implementation lane: BINARY. every visible case must pass, no cheating.
        all_cases_pass = True
        try:
            if exp_cases is not None and act_cases is not None:
                all_cases_pass = int(act_cases) >= int(exp_cases)
        except Exception:
            all_cases_pass = False
        if not all_cases_pass:
            hard.append("case_count_short")
        reward = 1.0 if (gates_ok and not hard and all_cases_pass) else 0.0
        if reward == 0.0 and not hard:
            hard.append("correctness_or_validation_incomplete")
        task_type = "implementation"
    else:
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            hard.append("invalid_primary_metric_value")
        # performance pre-gates (reward.md): all gates must pass AND speedup>1 AND ref>1.
        if not (math.isfinite(speedup) and math.isfinite(ref)):
            hard.append("non_finite_metric")
        elif speedup <= 1.0:
            hard.append("speedup_not_above_baseline")
        elif ref <= 1.0:
            hard.append("ref_speedup_invalid")
        if gates_ok and not hard:
            # log-shaped curve: speedup == ref -> 0.0 (must EXCEED it to score) ; speedup>=ref^2 -> 1.0
            reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))
            reward = max(0.0, min(1.0, reward))
        else:
            reward = 0.0
            if not hard:
                hard.append("validation_incomplete_or_bad_speedup")
        task_type = "performance"

    out = {
        "schema_version": "kernelbench_reward_v2_rewardmd",
        "task_type": task_type,
        "task_kind": task_kind,
        "reward": reward,
        "hard_fail_reasons": hard,
        "actual_hardware_type": state.get("actual_hardware_type", "") or bench.get("actual_hardware_type", ""),
        "state_path": str(STATE),
        "benchmark_path": str(BENCH),
    }
    if task_type == "performance":
        out["speedup"] = speedup
        out["ref_speedup"] = ref
        cv = bench.get("cv")
        if not isinstance(cv, dict):
            cv = {"baseline": bench.get("baseline_cv"), "candidate": bench.get("candidate_cv")}
        out["cv"] = cv
    else:
        try:
            total = int(exp_cases) if exp_cases is not None else 0
            passed = int(act_cases) if act_cases is not None else 0
        except Exception:
            total = passed = 0
        out["tests"] = {"passed": passed, "total": total}
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
