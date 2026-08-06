#!/usr/bin/env python3
"""FROZEN — e2e-b1-kv-traffic-sol reward driver (reviewer-owned, uploaded fresh at scoring).

Reward per the AUTHORITATIVE spec (the bench reward spec, 性能类 / performance class):

    reward = min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0          range [0.0, 1.0]

    speedup       = candidate vs the STRONG BASELINE, measured by ABBA pairing:
                    baseline / candidate suites are run ALTERNATELY for >= 5 pairs in fresh
                    sub-processes; each pair yields ratio_i = geomean_cases(base_ms/cand_ms)
                    (identical to geomean_sol_fraction(cand) / geomean_sol_fraction(base),
                    because sol_fraction = T_SOL / t_step and T_SOL cancels per case);
                    speedup = median_i(ratio_i).
    ref_speedup   = the ORACLE's median ABBA speedup, CALIBRATED AT AUTHORING TIME and pinned as
                    a CONSTANT in the frozen manifest (`ref_speedup.value`).  The oracle is NOT
                    in the image and is NEVER run at scoring time.

    speedup == ref_speedup  -> reward 0.5        speedup >= ref_speedup^2 -> reward 1.0 (capped)

Pre-gates — ANY hit means reward = 0.0 and no speedup is computed (reward.md §性能类前置门):
    1. build / import_smoke / readiness_probe failure  (harness crash, entry contract, frozen
       surface incomplete)
    2. correctness suite: ANY case fails (bit-exact KV round-trip, poison full-write, no-alias,
       current-plan, page-addressing, pool/peak budget, case-count cross-check vs the manifest)
    3. cheating: implausible sol_fraction (harness 1.02 bound), degenerate identity paired
       ratios, frozen-surface tamper (test.sh sha gate)
    4. candidate diff touches forbidden_edit_paths (test.sh frozen-surface sha gate)
    5. speedup <= 1.0   (did not cross the baseline)
    6. ref_speedup missing / null / <= 1.0  (no valid reference => HARD FAIL 0 with a reason;
       never silently treated as 1.0)

Every number is measured by THIS process in FRESH sub-processes (§1 G2/G3): the candidate never
reports its own latency, byte count, bandwidth or memory use.  The strong baseline is re-measured
in the same session inside every ABBA pair, so the candidate/baseline anchor is never stale.

On EVERY hard-fail path the metric is zeroed in BOTH reward.json and benchmark_results.json;
raw timings move under `measured_but_void_diagnostics` so nothing that reads like a score
survives a hard fail.
"""
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = "/logs/verifier"
BENCH = os.path.join(HERE, "harness", "bench_kvtraffic.py")
BASELINE_IMPL = os.path.join(HERE, "harness", "baseline_kv_traffic.py")
HIDDEN_SUITE = os.path.join(HERE, "harness", "hidden_suite.json")
MANIFEST = os.path.join(HERE, "verifier-correctness-manifest.json")
SUBMISSION = os.environ.get("SUBMISSION_DIR", "/app/repo/submission")
CANDIDATE_IMPL = os.path.join(SUBMISSION, "kv_traffic.py")
BASELINE_DRIFT_FLAG = 0.30
TASK_ID = "e2e-b1-kv-traffic-sol"
TASK_TYPE = "performance"
# reward.md requires >= 5 ABBA pairs.  Read from the (reviewer-owned, solver-unwritable) manifest
# but NEVER below the spec floor of 5.
ABBA_PAIRS_FLOOR = 5
ABBA_MAX_SECONDS_DEFAULT = 9000
# A candidate this far below the baseline can never reach speedup > 1 (measured reward noise on
# this task is +-3.6%), and gate 5 zeroes it either way — so stop after the first pair instead of
# spending 8 more suite runs on a guaranteed zero.
ABBA_EARLY_ABORT_BELOW = 0.90


def _py():
    for c in ("/opt/kernelbench-venv/bin/python3", "/usr/bin/python3", sys.executable):
        if c and os.path.exists(c):
            return c
    return "python3"


def _manifest():
    try:
        if os.path.exists(MANIFEST):
            return json.load(open(MANIFEST))
    except Exception as e:  # noqa: BLE001
        print("[verifier] manifest read error %r" % e)
    return {}


def _run_suite(impl_path, suite_path, tag, timeout=5400, timed_only=False):
    """Run the frozen harness in a FRESH process (recompute-from-artifact, §1 G3)."""
    out = os.path.join(tempfile.gettempdir(), "kb_kvbench_%s_%d.json" % (tag, int(time.time())))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The baked system is importable through PYTHONPATH so that must survive, but the candidate
    # must not be able to shadow the harness with cwd or its own submission root.
    raw = env.get("PYTHONPATH", "")
    keep = [e for e in raw.split(os.pathsep)
            if e and e != "." and not os.path.abspath(e).startswith(os.path.abspath(SUBMISSION))]
    if keep:
        env["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        env.pop("PYTHONPATH", None)
    cmd = [_py(), BENCH, "--impl", impl_path, "--suite", suite_path, "--out", out]
    if timed_only:
        cmd.append("--timed-only")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                           cwd="/tmp")
        tail = (p.stdout or "")[-4000:] + (p.stderr or "")[-4000:]
    except subprocess.TimeoutExpired:
        return {"status": "hard_fail", "reason": "%s: harness timeout after %ds" % (tag, timeout)}, ""
    if os.path.exists(out):
        with open(out) as fh:
            payload = json.load(fh)
        payload["wall_s"] = time.time() - t0
        return payload, tail
    return {"status": "hard_fail",
            "reason": "%s: harness produced no result (rc=%s)" % (tag, p.returncode)}, tail


def _cv(vals):
    vals = [float(v) for v in vals if v is not None and float(v) > 0]
    if len(vals) < 2:
        return None
    m = statistics.fmean(vals)
    if m <= 0:
        return None
    return round(statistics.stdev(vals) / m, 5)


def _emit(reward, quality_ok, detail, cand=None, base=None, hard_fail_reasons=None,
          speedup=None, ref_speedup=None, cv=None, abba=None):
    os.makedirs(LOGDIR, exist_ok=True)
    reward = float(min(1.0, max(0.0, reward)))
    hard_fail_reasons = list(hard_fail_reasons or [])
    void = bool(hard_fail_reasons) or not quality_ok
    if void:
        reward = 0.0
    cand = cand or {}
    base = base or {}
    cv = cv or {}
    abba = abba or {}
    state = {"task_id": TASK_ID, "lane": "e2e_task", "task_type": TASK_TYPE,
             "metric": "log_speedup_vs_ref_speedup",
             "reward": reward, "quality_gate_passed": bool(quality_ok),
             "speedup": speedup, "ref_speedup": ref_speedup,
             "hard_fail_reasons": hard_fail_reasons,
             "detail": detail, "ts": time.time()}
    with open(os.path.join(LOGDIR, "verifier_state.json"), "w") as fh:
        json.dump(state, fh, indent=1)
    corr = {"quality_gate": "bit_exact_kv_roundtrip + poison_full_write + no_alias + "
                            "current_plan + pool_and_peak_budget",
            "passed": bool(quality_ok), "detail": detail,
            "expected_case_count": cand.get("expected_case_count"),
            "observed_case_count": cand.get("observed_case_count"),
            "expected_correctness_case_count": cand.get("expected_correctness_case_count"),
            "observed_correctness_case_count": cand.get("observed_correctness_case_count"),
            "correctness_cases": cand.get("correctness_cases", []),
            "per_case": [{k: c.get(k) for k in
                          ("case_id", "op", "sol_fraction", "achieved_gbps", "bytes_min",
                           "pool_bytes", "pool_bytes_nominal", "peak_bytes",
                           "step_time_spread_pct") if k in c}
                         for c in cand.get("cases", [])]}
    with open(os.path.join(LOGDIR, "correctness_results.json"), "w") as fh:
        json.dump(corr, fh, indent=1)
    bench = {"metric": "log_speedup_vs_ref_speedup",
             "reward_formula": "min(1.0, ln(speedup/ref_speedup)/ln(ref_speedup)) if speedup > ref_speedup else 0.0",
             "peak_hbm_gbps_measured": cand.get("peak_hbm_gbps") or base.get("peak_hbm_gbps"),
             "reward": reward, "speedup": speedup, "ref_speedup": ref_speedup, "cv": cv,
             "abba": abba,
             "metric_void_on_hard_fail": bool(void)}
    payload = {"candidate_geomean_sol_fraction": cand.get("geomean_sol_fraction"),
               "strong_baseline_geomean_sol_fraction": base.get("geomean_sol_fraction"),
               "candidate_cases": cand.get("cases", []),
               "baseline_cases": base.get("cases", [])}
    if void:
        bench["speedup"] = 0.0
        bench["candidate_geomean_sol_fraction"] = 0.0
        bench["strong_baseline_geomean_sol_fraction"] = 0.0
        bench["measured_but_void_diagnostics"] = dict(payload)
        bench["measured_but_void_diagnostics"]["speedup_measured"] = speedup
        bench["measured_but_void_diagnostics"]["NOTE"] = (
            "NOT a score: the run hard-failed or missed the correctness gate; these raw "
            "measurements are diagnostics only and the metric is void.")
    else:
        bench.update(payload)
    with open(os.path.join(LOGDIR, "benchmark_results.json"), "w") as fh:
        json.dump(bench, fh, indent=1)
    # 🔴 The authoritative result JSON shape (reward.md §结果 JSON).
    with open(os.path.join(LOGDIR, "reward.json"), "w") as fh:
        json.dump({"task_type": TASK_TYPE, "reward": reward,
                   "hard_fail_reasons": hard_fail_reasons,
                   "speedup": speedup, "ref_speedup": ref_speedup, "cv": cv,
                   "quality_gate_passed": bool(quality_ok), "detail": detail}, fh, indent=1)
    with open(os.path.join(LOGDIR, "reward.txt"), "w") as fh:
        fh.write("%.6f\n" % reward)
    # 🔴 In LOOP mode (per-round dev scoring) this script's WHOLE stdout is teed by
    #    /opt/loop/score_engine.sh into /logs/loop/dev/verdict.raw, which the SOLVER can
    #    read. `detail` names the candidate AND baseline metric (the calibrated anchor and
    #    the hardware-relative internals are deliberately not disclosed), and a gate_fail
    #    detail names the hidden case id. So in LOOP mode print only reward + gate verdict.
    if os.environ.get("LOOP_DEV_OUT"):
        print("[verifier] reward=%.6f quality_ok=%s" % (reward, quality_ok))
    else:
        print("[verifier] reward=%.6f quality_ok=%s speedup=%s ref_speedup=%s :: %s"
              % (reward, quality_ok, speedup, ref_speedup, detail))


def _ref_speedup(man):
    """Read the AUTHORING-TIME calibrated constant. Never runs the oracle."""
    node = man.get("ref_speedup")
    if isinstance(node, dict):
        val, status = node.get("value"), node.get("status")
    else:
        val, status = node, None
    if val is None:
        return None, ("ref_speedup_not_calibrated: the frozen manifest carries no calibrated "
                      "ref_speedup constant (status=%s). reward.md requires reward = "
                      "ln(speedup/ref_speedup)/ln(ref_speedup); with no reference the task cannot be "
                      "scored and is a HARD FAIL (never silently treated as 1.0)." % status)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None, "ref_speedup_invalid: manifest ref_speedup=%r is not a number" % (val,)
    if not math.isfinite(val) or val <= 1.0:
        return None, ("ref_speedup_invalid: %.6f <= 1.0 — the reference solution does not beat "
                      "the strong baseline, so the task itself is not a valid performance "
                      "task (reward.md 前置门 6)" % val)
    return val, None


def _fail_reason(payload):
    """Map a harness failure onto a reward.md 前置门 reason.  `harness_tamper_detected` is a
    CHEATING finding (前置门 3), not a build failure: the candidate replaced a timing or
    comparison primitive of the bench harness in-process (MEASURED 2026-07-27, vacuity sweep V6)."""
    if "harness_tamper_detected" in str(payload.get("reason") or ""):
        return "cheating_harness_tamper"
    return ("correctness_failed" if payload.get("status") == "gate_fail"
            else "build_or_entry_contract_failed")


def _log_reward(speedup, ref_speedup):
    return min(1.0, max(0.0, min(1.0, math.log(speedup) / math.log(ref_speedup) - 1.0)))


def main():
    man = _manifest()
    mode = os.environ.get("VERIFIER_MODE", "candidate")
    impl = CANDIDATE_IMPL
    if mode == "strong_baseline":
        impl = BASELINE_IMPL
    elif mode.startswith("negative") or mode.startswith("ceiling"):
        alt = os.path.join(os.environ.get("NEGATIVE_DIR", "/opt/negative"),
                           mode.split("_", 1)[-1] + ".py")
        if os.path.exists(alt):
            impl = alt
    if not os.path.exists(impl):
        _emit(0.0, False, "entry contract: %s missing" % impl,
              hard_fail_reasons=["build_or_entry_contract_failed"])
        return 0
    for req in (BENCH, BASELINE_IMPL, HIDDEN_SUITE):
        if not os.path.exists(req):
            _emit(0.0, False, "frozen surface incomplete: %s missing" % req,
                  hard_fail_reasons=["build_or_entry_contract_failed"])
            return 0

    # ---- gate 6 first: no valid reference => hard fail before anything is measured. ----
    ref_speedup, ref_err = _ref_speedup(man)
    if ref_speedup is None:
        _emit(0.0, False, ref_err, hard_fail_reasons=["ref_speedup_invalid_or_missing"],
              ref_speedup=None)
        return 0

    # ---- gates 1-2: FULL candidate pass (correctness suite + timed cases). ----
    cand, ctail = _run_suite(impl, HIDDEN_SUITE, "cand")
    if cand.get("status") != "ok":
        reason = _fail_reason(cand)
        _emit(0.0, False, "candidate %s: %s" % (cand.get("status"), cand.get("reason")), cand,
              hard_fail_reasons=[reason], ref_speedup=ref_speedup)
        if not os.environ.get("LOOP_DEV_OUT"):
            print(ctail[-2500:])
        return 0
    for k_exp, k_obs, what in (("expected_case_count", "observed_case_count", "timed"),
                               ("expected_correctness_case_count",
                                "observed_correctness_case_count", "correctness")):
        if cand.get(k_exp) != cand.get(k_obs):
            _emit(0.0, False, "%s case count mismatch: expected %s observed %s"
                  % (what, cand.get(k_exp), cand.get(k_obs)), cand,
                  hard_fail_reasons=["correctness_failed"], ref_speedup=ref_speedup)
            return 0
    for key, obs in (("expected_case_count", cand.get("observed_case_count")),
                     ("expected_correctness_case_count",
                      cand.get("observed_correctness_case_count"))):
        pinned = man.get(key)
        if pinned is not None and int(pinned) != int(obs or -1):
            _emit(0.0, False, "suite tampered: manifest pins %s=%s, harness ran %s"
                  % (key, pinned, obs), cand, hard_fail_reasons=["forbidden_edit_path"],
                  ref_speedup=ref_speedup)
            return 0

    # ---- ABBA pairing (reward.md: baseline/candidate ALTERNATED, >= 5 pairs, median ratio). ----
    n_pairs = max(ABBA_PAIRS_FLOOR, int(man.get("abba_pairs") or ABBA_PAIRS_FLOOR))
    budget_s = float(man.get("abba_max_seconds") or ABBA_MAX_SECONDS_DEFAULT)
    t_start = time.time()
    ratios, base_geos, cand_geos = [], [], []
    base_last, warnings = {}, []
    # pair 0 reuses the FULL candidate pass above; every later pair re-times both sides.
    for i in range(n_pairs):
        # alternate the within-pair order so a monotone drift cannot bias one side: BA / AB / BA…
        base_first = (i % 2 == 0)
        if base_first:
            base, btail = _run_suite(BASELINE_IMPL, HIDDEN_SUITE, "base%d" % i, timed_only=True)
            if base.get("status") != "ok":
                _emit(0.0, False, "STRONG BASELINE failed to run (%s: %s) — verifier/baseline "
                                  "defect, not a candidate fault"
                      % (base.get("status"), base.get("reason")), cand, base,
                      hard_fail_reasons=["build_or_entry_contract_failed"],
                      ref_speedup=ref_speedup)
                if not os.environ.get("LOOP_DEV_OUT"):
                    print(btail[-2500:])
                return 0
            c_i = cand if i == 0 else None
            if c_i is None:
                c_i, ctail_i = _run_suite(impl, HIDDEN_SUITE, "cand%d" % i, timed_only=True)
        else:
            c_i, ctail_i = _run_suite(impl, HIDDEN_SUITE, "cand%d" % i, timed_only=True)
            base, btail = _run_suite(BASELINE_IMPL, HIDDEN_SUITE, "base%d" % i, timed_only=True)
            if base.get("status") != "ok":
                _emit(0.0, False, "STRONG BASELINE failed to run (%s: %s) — verifier/baseline "
                                  "defect, not a candidate fault"
                      % (base.get("status"), base.get("reason")), cand, base,
                      hard_fail_reasons=["build_or_entry_contract_failed"],
                      ref_speedup=ref_speedup)
                return 0
        if c_i.get("status") != "ok":
            reason = _fail_reason(c_i)
            _emit(0.0, False, "candidate repeat %d %s: %s"
                  % (i, c_i.get("status"), c_i.get("reason")), cand, base,
                  hard_fail_reasons=[reason], ref_speedup=ref_speedup)
            return 0
        bg, cg = float(base["geomean_sol_fraction"]), float(c_i["geomean_sol_fraction"])
        if bg <= 0:
            _emit(0.0, False, "baseline geomean sol_fraction %.6g <= 0" % bg, cand, base,
                  hard_fail_reasons=["build_or_entry_contract_failed"], ref_speedup=ref_speedup)
            return 0
        base_geos.append(bg)
        cand_geos.append(cg)
        # sol_fraction = T_SOL/t_step with an identical per-case T_SOL on both sides, so
        # cg/bg IS geomean_cases(baseline_ms / candidate_ms).
        ratios.append(cg / bg)
        base_last = base
        if i == 0 and ratios[0] < ABBA_EARLY_ABORT_BELOW:
            warnings.append("early_abort_after_1_pair: pair-0 speedup %.4f < %.2f, gate 5 "
                            "(speedup <= 1) is certain" % (ratios[0], ABBA_EARLY_ABORT_BELOW))
            break
        if i + 1 < n_pairs:
            elapsed = time.time() - t_start
            per_pair = elapsed / (i + 1)
            if elapsed + per_pair > budget_s:
                warnings.append("abba_budget_exhausted after %d pairs (%.0fs of %.0fs budget)"
                                % (i + 1, elapsed, budget_s))
                break

    speedup = statistics.median(ratios)
    abba = {"pairs_requested": n_pairs, "pairs_completed": len(ratios),
            "pair_ratios": [round(r, 6) for r in ratios],
            "order": "alternating baseline/candidate (BA, AB, BA, …)",
            "aggregation": "per pair: geomean over the 9 timed cases of baseline_ms/candidate_ms; "
                           "across pairs: median",
            "wall_s": round(time.time() - t_start, 1), "warnings": warnings}
    cv = {"baseline": _cv(base_geos), "candidate": _cv(cand_geos)}
    detail = ("speedup=%.5f (median of %d ABBA pairs %s) ref_speedup=%.5f"
              % (speedup, len(ratios), abba["pair_ratios"], ref_speedup))

    # ---- gate 3: degenerate identity paired ratios (a candidate that IS the baseline). ----
    if len(ratios) >= 2 and all(abs(r - 1.0) < 1e-9 for r in ratios):
        _emit(0.0, False, detail + " | identity paired ratios", cand, base_last,
              hard_fail_reasons=["cheating_identity_paired_ratios"], speedup=speedup,
              ref_speedup=ref_speedup, cv=cv, abba=abba)
        return 0

    # baseline drift diagnostic against the calibrated absolute anchor
    anchor = ((man.get("strong_baseline") or {}).get("geomean_sol_fraction")
              or man.get("strong_baseline_geomean_sol_fraction"))
    if anchor:
        bg_mean = statistics.fmean(base_geos)
        drift = abs(bg_mean - float(anchor)) / float(anchor)
        detail += " | baseline drift vs calibrated %.1f%%" % (100 * drift)
        if drift > BASELINE_DRIFT_FLAG:
            detail += (" | FLAG: baseline drifted >%.0f%% — shared state may be poisoned"
                       % (100 * BASELINE_DRIFT_FLAG))

    # ---- gate 5: did not cross the strong baseline. ----
    if speedup <= 1.0:
        _emit(0.0, True, detail + " | speedup <= 1: did not cross the strong baseline",
              cand, base_last, hard_fail_reasons=["speedup_not_above_baseline"],
              speedup=speedup, ref_speedup=ref_speedup, cv=cv, abba=abba)
        return 0

    reward = _log_reward(speedup, ref_speedup)
    _emit(reward, True, detail, cand, base_last, speedup=speedup, ref_speedup=ref_speedup,
          cv=cv, abba=abba)
    return 0


if __name__ == "__main__":
    sys.exit(main())
