#!/usr/bin/env python3
"""Standalone workload for the vLLM V1 scheduler PriorityRequestQueue subsystem.

Drives the in-scope PriorityRequestQueue imported from the baked /app/repo vLLM
tree. The queue serves waiting requests by ascending (priority, arrival_time,
insertion-seq). This is HOST-LOGIC (CPU): correctness is graded as an EXACT
pop-order trace (0/1) against an INDEPENDENT pure-Python reference; timing is a
deterministic CPU-wall proxy of the scheduler's per-step enqueue/drain cost.

Correctness (MANY diverse scenarios): for each scenario we run a mixed sequence
of add / pop / peek / remove operations and record the exact request-id trace the
queue produces; it must match the reference trace element-for-element. Scenarios
vary in size, priority spread, arrival ordering, and remove patterns, including
adversarial ties (equal priority, equal arrival_time -> FIFO) and single-element
edge cases.

Timing: build a large waiting set then drain it (the regime where the naive
unsorted-list backing is O(n^2) via a linear min-scan per pop while the heap is
O(n log n)); repeated so the fast oracle accrues a stable wall. Emits
WRO_PQ_RESULT.

Usage: python3 workload.py {correctness|timing}
"""
import json
import random
import sys
import time

from vllm.v1.core.sched.wro_priority_request_queue import PriorityRequestQueue


class Req:
    """Duck-typed stand-in for a scheduler Request: only priority + arrival_time
    are read by the queue (bypasses the heavy real Request __init__)."""
    __slots__ = ("request_id", "priority", "arrival_time")

    def __init__(self, rid, priority, arrival_time):
        self.request_id = rid
        self.priority = priority
        self.arrival_time = arrival_time


def reference_trace(ops):
    """Independent reference: an explicit list kept in sorted (priority,
    arrival_time, seq) order; returns the trace of request_ids produced by the
    pop/peek operations (and applies add/remove)."""
    items = []  # (priority, arrival_time, seq, req)
    seq = 0
    trace = []
    for op in ops:
        kind = op[0]
        if kind == "add":
            r = op[1]
            items.append((r.priority, r.arrival_time, seq, r))
            seq += 1
        elif kind in ("pop", "peek"):
            if not items:
                trace.append(None)
                continue
            items.sort(key=lambda e: (e[0], e[1], e[2]))
            r = items[0][3]
            trace.append(r.request_id)
            if kind == "pop":
                items.pop(0)
        elif kind == "remove":
            r = op[1]
            items = [e for e in items if e[3] is not r]
    return trace


def run_scope_trace(ops):
    q = PriorityRequestQueue()
    trace = []
    for op in ops:
        kind = op[0]
        if kind == "add":
            q.add_request(op[1])
        elif kind == "pop":
            trace.append(q.pop_request().request_id if len(q) > 0 else None)
        elif kind == "peek":
            trace.append(q.peek_request().request_id if len(q) > 0 else None)
        elif kind == "remove":
            q.remove_request(op[1])
    return trace


def gen_scenario(rng, n, n_priorities, tie_heavy):
    """Build a mixed op sequence of ~2n ops (adds interleaved with pops/removes)."""
    reqs = []
    for i in range(n):
        pr = rng.randint(0, max(0, n_priorities - 1))
        # tie_heavy -> quantize arrival_time so ties on (priority, arrival) occur
        at = float(rng.randint(0, 3)) if tie_heavy else rng.random() * 1000.0
        reqs.append(Req(f"r{i}", pr, at))
    ops = []
    pending = list(reqs)
    rng.shuffle(pending)
    added = []
    while pending or added:
        roll = rng.random()
        if pending and (roll < 0.55 or not added):
            r = pending.pop()
            ops.append(("add", r))
            added.append(r)
        elif added and roll < 0.85:
            ops.append(("pop",))
            # mirror the reference's removal of the current min so state tracks
            added.sort(key=lambda r: (r.priority, r.arrival_time))
            added.pop(0)
        elif added:
            victim = added[rng.randrange(len(added))]
            ops.append(("remove", victim))
            added = [r for r in added if r is not victim]
        else:
            ops.append(("peek",))
    # drain the rest
    ops.append(("peek",))
    for _ in range(n):
        ops.append(("pop",))
    return ops


CORR_SCENARIOS = [
    dict(n=3, n_priorities=1, tie_heavy=False),
    dict(n=4, n_priorities=2, tie_heavy=False),
    dict(n=5, n_priorities=1, tie_heavy=True),
    dict(n=8, n_priorities=3, tie_heavy=False),
    dict(n=10, n_priorities=2, tie_heavy=True),
    dict(n=16, n_priorities=4, tie_heavy=False),
    dict(n=20, n_priorities=1, tie_heavy=False),
    dict(n=24, n_priorities=5, tie_heavy=True),
    dict(n=32, n_priorities=3, tie_heavy=False),
    dict(n=40, n_priorities=8, tie_heavy=False),
    dict(n=48, n_priorities=2, tie_heavy=True),
    dict(n=64, n_priorities=6, tie_heavy=False),
    dict(n=50, n_priorities=1, tie_heavy=True),
    dict(n=80, n_priorities=4, tie_heavy=False),
    dict(n=100, n_priorities=10, tie_heavy=False),
    dict(n=6, n_priorities=3, tie_heavy=True),
    dict(n=7, n_priorities=2, tie_heavy=False),
    dict(n=128, n_priorities=5, tie_heavy=True),
    dict(n=90, n_priorities=7, tie_heavy=False),
    dict(n=150, n_priorities=3, tie_heavy=False),
]


def correctness():
    n_pass = 0
    detail = {}
    for i, sc in enumerate(CORR_SCENARIOS):
        try:
            rng = random.Random(900 + i)
            ops = gen_scenario(rng, sc["n"], sc["n_priorities"], sc["tie_heavy"])
            got = run_scope_trace(ops)
            exp = reference_trace(ops)
            ok = (got == exp)
            n_pass += int(ok)
            detail[f"case{i}"] = {"n_ops": len(ops), "trace_len": len(exp), "passed": ok}
        except Exception as e:
            detail[f"case{i}"] = {"error": f"{type(e).__name__}: {str(e)[:120]}", "passed": False}
    total = len(CORR_SCENARIOS)
    frac = n_pass / total
    print("WRO_PQ_RESULT " + json.dumps(
        {"correctness_ok": (n_pass == total), "correctness_frac": round(frac, 4),
         "n_pass": n_pass, "n_total": total, "detail": detail}))


# Timing: enqueue N then drain, repeated R times so the fast heap accrues a
# stable wall while the naive O(n^2) drain stays within the exec budget.
TIMING_N = 4000
TIMING_REPEAT = 8


def _build_reqs(n, seed=7):
    rng = random.Random(seed)
    return [Req(f"r{i}", rng.randint(0, 32), rng.random() * 1e6) for i in range(n)]


def timing():
    reqs = _build_reqs(TIMING_N)
    t0 = time.perf_counter()
    for _ in range(TIMING_REPEAT):
        q = PriorityRequestQueue()
        for r in reqs:
            q.add_request(r)
        while len(q) > 0:
            q.pop_request()
    ms = (time.perf_counter() - t0) * 1e3 / TIMING_REPEAT
    print("WRO_PQ_RESULT " + json.dumps({"timing_ms": round(ms, 6)}))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        correctness()
    elif mode == "timing":
        timing()
    else:
        print("WRO_PQ_RESULT " + json.dumps({"error": f"unknown mode {mode}"}))
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
