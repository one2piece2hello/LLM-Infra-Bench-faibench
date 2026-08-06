#!/usr/bin/env python3
"""Canonical kernelbench reward writer (task-agnostic; do NOT special-case a task here).

Reward definition:

  performance (acceleration):
      reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
      -> matching the oracle (speedup == ref_speedup) scores 0.5
      -> reaching ref_speedup**2 caps at 1.0
      value range is strictly [0.0, 1.0]
      HARD PRE-GATES (any one hit => reward 0, the formula is never evaluated):
        build/import failure, ANY correctness case fail, cheating, forbidden
        edit path, speedup <= 1, ref_speedup <= 1.
  implementation (correctness):
      reward = 1.0 iff EVERY test case passes and no cheating, else 0.0 (binary).

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

# Every gate below is a HARD gate: build/import origin, correctness (visible AND
# hidden), baseline sanity, benchmark validity, anti-cheat / forbidden-path.
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


def _case_counts(state, bench):
    """(passed, total) test cases when the verifier reports them; else (None, None)."""
    exp = state.get("expected_case_count")
    act = state.get("actual_case_count")
    try:
        exp = int(exp); act = int(act)
    except Exception:
        return (None, None)
    if exp <= 0:
        return (None, None)
    return (act, exp)


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE, {})
    bench = read_json(BENCH, {})
    hard = list(state.get("hard_fail_reasons") or [])
    task_kind = state.get("task_kind") or bench.get("task_kind") or "acceleration"
    gates_ok = all(bool(state.get(g)) for g in GATES)

    passed, total = _case_counts(state, bench)
    # A short test suite is a correctness FAIL, not a partial credit: reward.md
    # requires EVERY case to pass (implementation) / no case to fail (performance).
    all_cases_ok = True
    if passed is not None and total is not None and passed < total:
        all_cases_ok = False
        if not any("case" in str(h) or "correctness" in str(h) for h in hard):
            hard.append("correctness_cases_incomplete")

    speedup = ref = 1.0
    reward = 0.0

    if task_kind == "correctness":
        # ---- implementation class: BINARY. All cases pass + no cheating -> 1.0, else 0.0.
        reward = 1.0 if (gates_ok and all_cases_ok and not hard) else 0.0
        if reward == 0.0 and not hard:
            hard.append("correctness_or_validation_incomplete")
    else:
        # ---- performance class: log-ratio reward, behind the hard pre-gates.
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            hard.append("invalid_primary_metric_value")

        if not (math.isfinite(speedup) and speedup > 0.0):
            hard.append("invalid_primary_metric_value")
        # pre-gate 5: the candidate must actually beat the baseline
        elif speedup <= 1.0:
            hard.append("speedup_not_above_baseline")
        # pre-gate 6: the reference anchor must be a real speedup
        if not (math.isfinite(ref) and ref > 1.0):
            hard.append("ref_speedup_invalid")

        if gates_ok and all_cases_ok and not hard:
            reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))
            # value range is strictly [0.0, 1.0]
            reward = max(0.0, min(1.0, reward))
        else:
            reward = 0.0
            if not hard:
                hard.append("validation_incomplete_or_bad_speedup")

    out = {
        "schema_version": "kernelbench_reward_v2_logratio",
        # reward.md result-JSON field: "performance" | "implementation"
        "task_type": "implementation" if task_kind == "correctness" else "performance",
        "task_kind": task_kind,
        "reward": reward,
        "speedup": speedup,
        "ref_speedup": ref,
        "reward_formula": ("binary: 1.0 iff all cases pass and no cheating, else 0.0"
                           if task_kind == "correctness"
                           else "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0; 0 if any hard pre-gate hit"),
        "actual_hardware_type": state.get("actual_hardware_type", "") or bench.get("actual_hardware_type", ""),
        "hard_fail_reasons": hard,
        "state_path": str(STATE),
        "benchmark_path": str(BENCH),
    }
    # reward.md result-JSON contract: performance always carries `cv`,
    # implementation always carries `tests{passed,total}`.
    if out["task_type"] == "performance":
        out["cv"] = bench.get("cv")          # None when this harness does not measure dispersion
        if passed is not None and total is not None:
            out["tests"] = {"passed": passed, "total": total}
    else:
        out["tests"] = ({"passed": passed, "total": total}
                        if (passed is not None and total is not None)
                        else {"passed": 0 if hard else 1, "total": 1})
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
