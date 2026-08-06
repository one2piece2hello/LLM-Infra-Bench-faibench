"""compute_reward.py -- reward writer for wro-memory-accounting-sim-coupled (reformed 2026-07-27, group G1).

Log-ratio PERFORMANCE reward (see the bench reward spec, the authoritative
spec for this bench):

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0

  speedup     = baseline_ms / candidate_ms      (this run's candidate vs. the
                degraded baseline restored at HEAD by test.sh)
  ref_speedup = the oracle's median speedup within the same image (baked
                calibration manifest / WRO_REF)
  speedup == ref_speedup    -> reward = 0.0  (must EXCEED the oracle)
  speedup >= ref_speedup**2 -> reward capped at 1.0

Front gates -- ANY hit means reward = 0 and the log-ratio is never evaluated:
  1. build/import/readiness failure   (WRO_HARD, scope_ok, import_origin_ok, benchmark_ok)
  2. correctness fails ANY case       (correctness_frac < 1.0)
  3. cheating / out-of-scope edit     (folded into WRO_HARD upstream by test.sh:
                                        scope-diff gate, import-origin assert)
  4. forbidden_edit_paths touched     (same upstream out_of_scope_edit gate)
  5. speedup <= 1                     (candidate did not beat the baseline)
  6. ref_speedup <= 1                 (oracle calibration itself invalid)

Task context: the reference (oracle) solution rewrites the in-scope pair
memsim/accountant.py + memsim/scheduler.py, and `ref_speedup` is that solution's
calibrated speedup inside this image. The oracle's technique is deliberately NOT
described in this file -- it is baked into the task container at
/opt/loop/private/tests/ and is therefore readable by the solver. (The previous
compute_reward.py in this file carried a "wro-flamegraph-fold-tree-coupled"
docstring header -- a copy-paste artifact from an unrelated task; corrected
here.)

Removed vs. the old gated_oracle template: the `vs_oracle > 2.0 -> hard_fail
implausible_vs_oracle` guard. Under the OLD uncapped formula (0.5 + 0.5*vs_oracle)
that guard was the ONLY thing bounding the score, so it had to hard-zero anything
past 2x-the-oracle. The NEW formula's min(1.0, ...) is itself a hard ceiling, so a
separate zeroing guard on the same ratio is redundant -- and worse, under the new
formula it would arbitrarily zero a *legitimately* excellent solution (anything
scoring 1.0 already sits at speedup >= ref_speedup**2, often >> 2x). The guard was
never independent cheat-detection: it never re-ran correctness or re-verified
anything, it only thresholded the same base_ms/cand_ms pair already used for
scoring. Real anti-cheat coverage (scope-diff gate, import-origin assert, the
correctness suite itself) lives entirely upstream in test.sh and is unchanged by
this removal. A non-gating diagnostic ratio is kept below (`vs_oracle`) purely
for human audit visibility.

cv (ABBA coefficient-of-variation): reward.md's result-JSON schema includes a `cv`
field from >=5 interleaved ABBA baseline/candidate pairs. This task's test.sh only
takes ONE candidate-timing and ONE baseline-timing sample (a single git-stash/
checkout dance around one `workload.py timing` call each), not a true ABBA loop --
a pre-existing structural gap in test.sh/workload.py, out of scope for this reward-
formula reform. `cv` is therefore emitted as null with an explanatory note rather
than a fabricated number.
"""
import json
import math
import os


def f(x, d=-1.0):
    try:
        return float(x)
    except Exception:
        return d


LOG = os.environ.get("WRO_LOG", "/logs/verifier")
os.makedirs(LOG, exist_ok=True)

mode = os.environ.get("WRO_MODE", "candidate")
hard = [h for h in os.environ.get("WRO_HARD", "").split() if h]
corr_frac = f(os.environ.get("WRO_CORR_FRAC", os.environ.get("WRO_FRAC", "0")))
base_ms = f(os.environ.get("WRO_BASE_MS", "-1"))
cand_ms = f(os.environ.get("WRO_CAND_MS", "-1"))
ref_speedup = f(os.environ.get("WRO_REF", "1"), 1.0)

scope_ok = os.environ.get("WRO_SCOPE") == "true"
import_ok = os.environ.get("WRO_IMP") == "true"
bench_ok = os.environ.get("WRO_BENCH") == "true"

gates = {
    "scope_ok": scope_ok,
    "import_origin_ok": import_ok,
    "correctness_ok": corr_frac >= 0.999,
    "benchmark_ok": bench_ok,
}
if "WRO_BAN" in os.environ:
    gates["vendor_op_ban_ok"] = os.environ.get("WRO_BAN") == "true"
if "WRO_CORR" in os.environ:
    gates["correctness_flag_ok"] = os.environ.get("WRO_CORR") == "true"

speedup = (base_ms / cand_ms) if (base_ms > 0 and cand_ms > 0) else 0.0
vs_oracle = (speedup / ref_speedup) if ref_speedup > 0 else 0.0  # diagnostic only, not gated on

hard_fail_reasons = list(hard)
if corr_frac < 0.999:
    hard_fail_reasons.append("correctness_frac_below_1:%.6f" % corr_frac)
if speedup <= 1.0:
    hard_fail_reasons.append("speedup_not_above_baseline:%.6f" % speedup)
if ref_speedup <= 1.0:
    hard_fail_reasons.append("ref_speedup_invalid:%.6f" % ref_speedup)

if hard_fail_reasons:
    reward = 0.0
else:
    try:
        reward = min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))
    except (ValueError, ZeroDivisionError):
        reward = 0.0
        hard_fail_reasons.append("log_ratio_undefined")
reward = round(max(0.0, reward), 6)

verifier_state = {
    "mode": mode,
    "hard_fails": hard_fail_reasons,
    "gates": gates,
    "ok": not hard_fail_reasons,
}
correctness_results = {"correctness_frac": corr_frac, "passed": corr_frac >= 0.999}
benchmark_results = {
    "baseline_ms": base_ms,
    "candidate_ms": cand_ms,
    "speedup": round(speedup, 6),
    "ref_speedup": ref_speedup,
    "perf_metric": "speedup",
    "metadata": {"vs_oracle": round(vs_oracle, 6)},
}
reward_json = {
    "task_type": "performance",
    "reward": reward,
    "score": reward,
    "mode": mode,
    "hard_fail_reasons": hard_fail_reasons,
    "hard_fails": hard_fail_reasons,
    "speedup": round(speedup, 6),
    "ref_speedup": ref_speedup,
    "vs_oracle": round(vs_oracle, 6),
    "correctness_frac": corr_frac,
    "cv": {"baseline": None, "candidate": None},
    "cv_note": "single-shot timing (no ABBA repeats) in current test.sh/workload.py; cv not computable -- pre-existing structural gap, out of scope for this reward-formula reform",
    "gates": gates,
}

with open(os.path.join(LOG, "verifier_state.json"), "w") as fh:
    json.dump(verifier_state, fh)
with open(os.path.join(LOG, "correctness_results.json"), "w") as fh:
    json.dump(correctness_results, fh)
with open(os.path.join(LOG, "benchmark_results.json"), "w") as fh:
    json.dump(benchmark_results, fh)
with open(os.path.join(LOG, "reward.json"), "w") as fh:
    json.dump(reward_json, fh)
with open(os.path.join(LOG, "reward.txt"), "w") as fh:
    fh.write(str(reward))

# the line the scorer parses: last stdout JSON object
# (must contain at least "reward"; "gates" + "hard_fails" kept for back-compat)
print(json.dumps(reward_json))
