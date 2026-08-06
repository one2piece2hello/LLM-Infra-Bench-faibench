#!/usr/bin/env python3
"""Canonical kernelbench reward writer.

Dual-mode:
  - implementation / correctness task (verifier_state.task_kind == "correctness"):
        BINARY. reward = 1.0 iff every hard gate is true, there are no
        hard_fail_reasons and no visible case failed; otherwise 0.0.
  - performance / acceleration task:
        reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0
        (log curve: speedup == ref_speedup -> 0.00; speedup >= ref_speedup**2 -> 1.0)
        Pre-gates that force reward = 0 WITHOUT entering the formula:
          build/import/verifier gate failure, any correctness case fail, anti-cheat
          or forbidden-edit-path hit, speedup <= 1, ref_speedup <= 1.

Task-agnostic; do NOT special-case a task here. `ref_speedup` is read from the
benchmark payload exactly as before (tests/ref_speedup.txt or the in-image
manifest feed it upstream) — this file never invents or rescales it.

Reads:  /logs/verifier/verifier_state.json, /logs/verifier/benchmark_results.json,
        /logs/verifier/correctness_results.json (optional, for the tests{passed,total} block)
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


def _case_counts(state, corr):
    """(passed, total) over the visible/hidden case suite; -1 when unknown."""
    total = state.get("expected_case_count")
    passed = state.get("actual_case_count")
    if total is None or passed is None:
        try:
            suites = corr.get("suites") or []
            total = sum(int(s.get("expected_case_count") or 0) for s in suites)
            passed = sum(int(s.get("passed") or 0) for s in suites)
        except Exception:
            total = passed = None
    try:
        return int(passed), int(total)
    except Exception:
        return -1, -1


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE, {})
    bench = read_json(BENCH, {})
    corr = read_json(CORR, {})
    hard = list(state.get("hard_fail_reasons") or [])
    task_kind = state.get("task_kind") or bench.get("task_kind") or "acceleration"
    gates_ok = all(bool(state.get(g)) for g in GATES)
    passed, total = _case_counts(state, corr)
    cases_ok = True if (passed < 0 or total < 0) else (passed >= total)

    speedup = ref = 1.0
    reward = 0.0
    if task_kind == "correctness":
        # IMPLEMENTATION lane: strictly BINARY. Every case must pass, every gate
        # must hold and no anti-cheat / forbidden-path reason may be present.
        task_type = "implementation"
        if gates_ok and not hard and cases_ok:
            reward = 1.0
        else:
            reward = 0.0
            if not hard:
                hard.append("case_failed_or_validation_incomplete" if not cases_ok
                            else "correctness_or_validation_incomplete")
    else:
        task_type = "performance"
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            hard.append("invalid_primary_metric_value")
        # --- pre-gates (reward.md 性能类前置门): any hit -> 0, formula not entered ---
        if not gates_ok or not cases_ok:
            reward = 0.0
            if not hard:
                hard.append("case_failed_or_validation_incomplete" if not cases_ok
                            else "validation_incomplete")
        elif hard:
            reward = 0.0
        elif not (math.isfinite(speedup) and speedup > 1.0):
            reward = 0.0
            hard.append("speedup_not_above_baseline")
        elif not (math.isfinite(ref) and ref > 1.0):
            reward = 0.0
            hard.append("ref_speedup_invalid")
        else:
            # log curve, clamped to [0.0, 1.0]
            reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref) - 1.0)))
            reward = max(0.0, reward)

    out = {
        "schema_version": "kernelbench_reward_v3_oracle_relative",
        "task_type": task_type,
        "task_kind": task_kind,
        "reward": reward,
        "reward_formula": ("min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0" if task_type == "performance"
                           else "1.0 iff all cases pass and no cheat, else 0.0"),
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
        out["tests"] = {"passed": passed, "total": total}
    else:
        out["tests"] = {"passed": passed, "total": total}
        out["speedup"] = speedup
        out["ref_speedup"] = ref
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
