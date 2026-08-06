#!/usr/bin/env python3
"""Canonical kernelbench reward writer (reused verbatim from the ready-task
template kernel-opt-p0a4-003). Task-agnostic; do NOT special-case a task here.

Dual-mode:
  - correctness task (verifier_state.task_kind == "correctness"):
        reward = 1.0 iff all hard gates true and no hard_fail_reasons, else 0.0; no timing.
  - acceleration task:
        reward = speedup (raw measured speedup — no cap, no log, no ref anchor)
        iff all gates pass and speedup is finite and > 0; else 0.0.

Reads:  /logs/loop/dev/verifier_state.json, /logs/loop/dev/benchmark_results.json
Writes: /logs/loop/dev/reward.json, /logs/loop/dev/reward.txt
"""
import json, math
from pathlib import Path

LOG = Path("/logs/loop/dev")
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

    speedup = ref = 1.0
    reward = 0.0
    if task_kind == "correctness":
        reward = 1.0 if (gates_ok and not hard) else 0.0
        if reward == 0.0 and not hard:
            hard.append("correctness_or_validation_incomplete")
    else:
        try:
            speedup = float(bench.get("aggregate_speedup") or bench.get("primary_metric_value") or 1.0)
            ref = float(bench.get("ref_speedup") or bench.get("reference_metric_value") or 1.0)
        except Exception:
            speedup = ref = 1.0
            hard.append("invalid_primary_metric_value")
        # reward IS the raw measured speedup (no cap, no log, no ref anchor);
        # ref_speedup is kept as calibration metadata only. reward is 0 only when a
        # gate fails (correctness/anti-cheat) or the measured speedup is unusable.
        if gates_ok and not hard and math.isfinite(speedup) and speedup > 0.0:
            reward = speedup
        if reward == 0.0 and not hard:
            hard.append("validation_incomplete_or_bad_speedup")

    out = {
        "schema_version": "kernelbench_reward_v1",
        "task_kind": task_kind,
        "reward": reward,
        "speedup": speedup,
        "ref_speedup": ref,
        "actual_hardware_type": state.get("actual_hardware_type", "") or bench.get("actual_hardware_type", ""),
        "hard_fail_reasons": hard,
        "state_path": str(STATE),
        "benchmark_path": str(BENCH),
    }
    REWARD.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TEXT.write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
