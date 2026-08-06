#!/usr/bin/env python3
"""compute_reward.py — read verify_core's result json, emit the 5-file verifier output contract
under /logs/verifier.

reward (IMPLEMENTATION class -> BINARY):
    reward = 1.0 iff correctness passed (every hidden case) and no hard fail;
    reward = 0.0 otherwise.
verify_core.py is the authority and already applies this rule; this module mirrors it and
re-derives the binary value defensively. vs_oracle / speedup are carried through as
DIAGNOSTIC METADATA only, never as the reward.
"""
import json
import os
import sys


def main():
    res_path = sys.argv[1] if len(sys.argv) > 1 else "/logs/verifier/wre_result.json"
    outdir = os.environ.get("VERIFIER_OUT_DIR", "/logs/verifier")
    os.makedirs(outdir, exist_ok=True)
    try:
        with open(res_path) as f:
            r = json.load(f)
    except Exception as e:
        r = {"reward": 0.0, "correctness_passed": False, "reason": f"no_result:{e}"}

    correctness_ok = bool(r.get("correctness_passed", False))
    hard = list(r.get("hard_fail_reasons") or [])
    if not correctness_ok and not hard:
        hard = ["correctness_failed"]
    # BINARY: all cases pass and no hard fail => 1.0, else 0.0.
    reward = 1.0 if (correctness_ok and not hard) else 0.0
    speedup = float(r.get("speedup", 0.0) or 0.0)
    tests = r.get("tests") or {"passed": r.get("n_checks") if correctness_ok else 0,
                               "total": r.get("n_checks")}

    verifier_state = {"status": "completed", "mode": r.get("mode", "candidate"),
                      "task_kind": "correctness", "correctness_ok": correctness_ok,
                      "hard_fail_reasons": hard}
    correctness_results = {"passed": correctness_ok, "n_checks": r.get("n_checks"),
                           "failed_checks": r.get("failed_checks", []), "reason": r.get("reason", "")}
    benchmark_results = {"perf_metric": "vs_oracle (metadata only)", "vs_oracle": speedup,
                         "speedup": speedup,
                         "candidate_ms": r.get("candidate_ms"), "oracle_ms": r.get("oracle_ms"),
                         "latency_stability": r.get("latency_stability"),
                         "note": "timing is diagnostic metadata; the reward is binary"}
    reward_json = {"task_type": "implementation", "reward": reward,
                   "reward_formula": "implementation: 1.0 iff every case passes and no cheat, else 0.0",
                   "hard_fail_reasons": hard, "tests": tests,
                   "metadata": {"vs_oracle": speedup,
                                "candidate_ms": r.get("candidate_ms"),
                                "oracle_ms": r.get("oracle_ms"),
                                "note": "vs_oracle is diagnostic metadata only, never the reward"},
                   "speedup": speedup}

    for name, obj in [("verifier_state.json", verifier_state),
                      ("correctness_results.json", correctness_results),
                      ("benchmark_results.json", benchmark_results),
                      ("reward.json", reward_json)]:
        with open(os.path.join(outdir, name), "w") as f:
            json.dump(obj, f)
    with open(os.path.join(outdir, "reward.txt"), "w") as f:
        f.write(f"{reward:.6f}\n")
    print(json.dumps(reward_json))


if __name__ == "__main__":
    main()
