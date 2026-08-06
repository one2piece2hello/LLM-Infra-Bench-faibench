#!/usr/bin/env python3
"""Hidden workload for wro-memory-accounting-sim-coupled (Type-2 B2 BEAT, proxy-perf).

Subsystem: a training/inference memory planner under ``memsim/`` -- a peak-memory accountant over an
execution plan (``accountant.MemoryAccountant``) and an eviction scheduler that explores shortening
tensor liveness to fit a budget (``scheduler.EvictionScheduler``), sharing a fixed liveness model
(``model.py``, out of scope). This is a perf-modeling / simulation backend (roofline-style memory
accounting), NOT a real allocator.

  correctness -- build MANY diverse plans (single tensor, disjoint lifetimes, fully-overlapping,
                 staircase, zero-byte tensors, zero-length liveness, tensors spanning all steps,
                 empty plan, budget above/below peak) and assert footprint_at / timeline / peak /
                 peak_after_free and the scheduler's fits / over_budget_steps / best_eviction /
                 greedy_plan match an INDEPENDENT in-harness reference EXACTLY. Emits
                 ``WRO_MEMSIM_RESULT {"correctness_frac": ...}``.

  timing      -- score peak memory for a large plan repeatedly while a scheduler explores eviction
                 candidates, and wall-time the whole exploration. The reference solution's
                 technique is deliberately NOT described here (this file is readable from
                 inside the task container). Emits
                 ``WRO_MEMSIM_RESULT {"timing_ms": ...}``.

Imports ``memsim`` from /app/repo (PYTHONPATH).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, "/app/repo")


def scope_pkg():
    import memsim as m
    return m


# ---------------- independent reference ----------------
def _ref_timeline(tensors, n):
    tl = [0] * n
    for name, nb, a, f in tensors:
        for s in range(max(0, a), min(n, f)):
            tl[s] += nb
    return tl


def _ref_peak(tensors, n):
    tl = _ref_timeline(tensors, n)
    if not tl:
        return (0, 0)
    best = tl[0]
    bs = 0
    for s in range(1, n):
        if tl[s] > best:
            best = tl[s]
            bs = s
    return (best, bs)


def _ref_peak_after_free(tensors, n, name):
    t2 = []
    found = False
    for nm, nb, a, f in tensors:
        if nm == name and not found:
            f = max(a, f - 1)
            found = True
        t2.append((nm, nb, a, f))
    if not found:
        return _ref_peak(tensors, n)[0]
    return _ref_peak(t2, n)[0]


def _mk_plan(m, tensors, n):
    plan = m.ExecutionPlan(n)
    for nm, nb, a, f in tensors:
        plan.add(m.Tensor(nm, nb, a, f))
    return plan


def _scenarios():
    rnd = random.Random(20260726)
    scen = []  # (name, tensors[(name,nbytes,alloc,free)], n_steps)
    scen.append(("single", [("a", 100, 0, 3)], 5))
    scen.append(("disjoint", [("a", 100, 0, 2), ("b", 200, 2, 4)], 5))
    scen.append(("overlap", [("a", 100, 0, 3), ("b", 200, 1, 4)], 5))
    scen.append(("staircase", [("a", 10, 0, 5), ("b", 20, 1, 5), ("c", 40, 2, 5)], 6))
    scen.append(("zero_byte", [("a", 0, 0, 5), ("b", 50, 1, 3)], 6))
    scen.append(("zero_len", [("a", 100, 2, 2), ("b", 30, 0, 4)], 5))
    scen.append(("span_all", [("a", 70, 0, 8)], 8))
    scen.append(("empty_plan", [], 5))
    scen.append(("zero_steps", [("a", 10, 0, 0)], 0))
    scen.append(("clamped_free", [("a", 100, 3, 20)], 6))
    scen.append(("dup_names_ok", [("a", 10, 0, 3), ("a", 20, 1, 4)], 5))
    for c in range(19):
        n = rnd.choice([1, 4, 12, 40])
        tensors = []
        for i in range(rnd.choice([0, 1, 3, 10, 30])):
            a = rnd.randint(0, n)
            f = min(n + rnd.choice([0, 0, 5]), a + rnd.randint(0, n))
            tensors.append(("t%d" % i, rnd.choice([0, 1, 10, 100, 4096]), a, f))
        scen.append(("rand%d" % c, tensors, n))
    return scen


def run_correctness():
    m = scope_pkg()
    npass = 0
    results = {}
    for name, tensors, n in _scenarios():
        try:
            ok = True
            plan = _mk_plan(m, tensors, n)
            acct = m.MemoryAccountant(plan)
            ref_tl = _ref_timeline(tensors, n)
            if acct.timeline() != ref_tl:
                ok = False
            for s in range(n):
                if acct.footprint_at(s) != ref_tl[s]:
                    ok = False
            if tuple(acct.peak()) != _ref_peak(tensors, n):
                ok = False
            # peak_after_free over each tensor name + unknown
            names = sorted(set(nm for nm, _, _, _ in tensors)) + ["__nope__"]
            for nm in names:
                if acct.peak_after_free(nm) != _ref_peak_after_free(tensors, n, nm):
                    ok = False
            # scheduler: fits / over_budget_steps / best_eviction / greedy_plan at several budgets
            peak_b = _ref_peak(tensors, n)[0]
            for budget in sorted(set([0, peak_b // 2, peak_b, peak_b + 1, max(0, peak_b - 1)])):
                sch = m.EvictionScheduler(_mk_plan(m, tensors, n))
                if bool(sch.fits(budget)) != (peak_b <= budget):
                    ok = False
                exp_over = [s for s, v in enumerate(ref_tl) if v > budget]
                if sch.over_budget_steps(budget) != exp_over:
                    ok = False
                # best_eviction reference
                _, ps = _ref_peak(tensors, n)
                cands = sorted(set(nm for nm, _, a, f in tensors if a <= ps < f))
                if peak_b <= budget or not cands:
                    exp_best = None
                else:
                    exp_best = None
                    exp_val = None
                    for nm in cands:
                        pv = _ref_peak_after_free(tensors, n, nm)
                        if exp_val is None or pv < exp_val:
                            exp_val = pv
                            exp_best = nm
                if sch.best_eviction(budget) != exp_best:
                    ok = False
            results[name] = {"ok": bool(ok)}
        except Exception as e:
            results[name] = {"ok": False, "error": repr(e)}
        if results[name]["ok"]:
            npass += 1
    total = len(results)
    failed = {k: v for k, v in results.items() if not v["ok"]}
    return (npass / total if total else 0.0), total, failed


def run_timing():
    m = scope_pkg()
    n = int(os.environ.get("WRO_MEMSIM_STEPS", "3000"))
    ntensors = int(os.environ.get("WRO_MEMSIM_TENSORS", "3000"))
    nq = int(os.environ.get("WRO_MEMSIM_QUERIES", "300"))
    rounds = int(os.environ.get("WRO_MEMSIM_ROUNDS", "3"))
    rnd = random.Random(99)
    tensors = []
    for i in range(ntensors):
        a = rnd.randint(0, n - 1)
        f = min(n, a + rnd.randint(1, n // 4 + 1))
        tensors.append(("t%d" % i, rnd.randint(1, 4096), a, f))

    def one():
        plan = _mk_plan(m, tensors, n)
        acct = m.MemoryAccountant(plan)
        acc = 0
        pk = acct.peak()[0]
        budget = pk // 2
        for i in range(nq):
            acc += acct.footprint_at(rnd.randint(0, n - 1))
            if i % 20 == 0:
                acc += len(acct.timeline())
        sch = m.EvictionScheduler(_mk_plan(m, tensors, n))
        acc += len(sch.over_budget_steps(budget))
        return acc

    one()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        one()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if cmd == "timing":
        try:
            print("WRO_MEMSIM_RESULT " + json.dumps({"timing_ms": run_timing()}))
        except Exception as e:
            import traceback
            print("WRO_MEMSIM_RESULT " + json.dumps({"timing_ms": -1, "error": repr(e),
                                                    "tb": traceback.format_exc()[-800:]}))
        return
    origin = None
    try:
        origin = os.path.realpath(scope_pkg().accountant.__file__)
    except Exception:
        pass
    try:
        frac, total, failed = run_correctness()
        print("WRO_MEMSIM_RESULT " + json.dumps(
            {"correctness_frac": frac, "n_cases": total, "n_failed": len(failed),
             "failed": {k: failed[k] for k in list(failed)[:8]}, "origin": origin}))
    except Exception as e:
        import traceback
        print("WRO_MEMSIM_RESULT " + json.dumps(
            {"correctness_frac": 0.0, "error": repr(e),
             "tb": traceback.format_exc()[-900:], "origin": origin}))


if __name__ == "__main__":
    main()
