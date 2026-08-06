#!/usr/bin/env python3
"""Hidden workload for wro-causal-delivery-vclock-coupled (Type-2 B2 BEAT, proxy-perf).

Subsystem: a causal-broadcast delivery layer under ``causal/`` -- a pending-message store
(``buffer.PendingStore``) and the delivery driver (``channel.CausalChannel``), sharing a fixed vector
clock (``vclock.py``, out of scope). The channel delivers arriving messages in an order consistent
with the happens-before relation, buffering out-of-order arrivals until they are causally deliverable.

  correctness -- build MANY diverse arrival interleavings (in-order, fully reversed, interleaved
                 senders, single sender, concurrent messages, duplicate-free chains, deep dependency
                 chains, out-of-order bursts) and assert the DELIVERY ORDER (delivered_ids), the
                 per-arrival returned batches, pending_count, and the final clock all match an
                 INDEPENDENT in-harness reference EXACTLY. Emits ``WRO_CAUSAL_RESULT {"correctness_frac": ...}``.

  timing      -- feed a large stream where each sender's messages arrive in reverse order (so they
                 buffer, then cascade-deliver). The naive driver re-scans the whole pending buffer
                 after every delivery (O(N^2)); the indexed driver only re-checks per-sender
                 front-runners (O(N * senders)). Headroom GROWS with the number of buffered messages.
                 Emits ``WRO_CAUSAL_RESULT {"timing_ms": ...}``.

Imports ``causal`` from /app/repo (PYTHONPATH).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, "/app/repo")


def scope_pkg():
    import causal as m
    return m


# ---------------- independent reference (straightforward causal delivery) ----------------
def _ref_deliverable(pid, vc, D):
    if vc.get(pid, 0) != D.get(pid, 0) + 1:
        return False
    for k, v in vc.items():
        if k == pid:
            continue
        if v > D.get(k, 0):
            return False
    return True


class _RefChannel:
    def __init__(self):
        self.D = {}
        self.pending = []   # (mid, pid, vc_dict)
        self.delivered = []

    def deliver(self, msg):
        mid, pid, vc = msg
        self.pending.append((mid, pid, dict(vc)))
        out = []
        while True:
            best = None
            for (m2, p2, v2) in self.pending:
                if _ref_deliverable(p2, v2, self.D):
                    if best is None or m2 < best[0]:
                        best = (m2, p2, v2)
            if best is None:
                break
            m2, p2, v2 = best
            self.D[p2] = self.D.get(p2, 0) + 1
            self.delivered.append(m2)
            self.pending = [(a, b, c) for (a, b, c) in self.pending if a != m2]
            out.append(m2)
        return out

    def clock(self):
        return {p: v for p, v in self.D.items() if v != 0}


def _gen_causal_stream(rnd, n_senders, per_sender, arrival="reverse"):
    """Generate a valid causal-broadcast stream. Each sender p emits messages with strictly
    increasing local counts; a message's vc is the sender's clock after ticking, merged with the
    causal history it 'saw'. To keep the reference simple we use per-sender chains (each message
    depends only on the sender's previous message + optionally one cross-sender dependency already
    'seen'). Returns (arrival_list, msg_defs) where each msg = (mid, pid, vc_dict)."""
    msgs = []
    mid = 0
    # simple model: sender p's k-th message has vc = {p: k} plus dependencies on already-created msgs
    sender_last_vc = {p: {} for p in range(n_senders)}
    global_seen = {}
    for k in range(1, per_sender + 1):
        for p in range(n_senders):
            vc = dict(sender_last_vc[p])
            vc[p] = k
            # occasionally depend on the current global frontier (a cross-sender causal edge)
            if rnd.random() < 0.3:
                for q, c in global_seen.items():
                    if c > vc.get(q, 0):
                        vc[q] = c
            sender_last_vc[p] = dict(vc)
            msgs.append((mid, p, vc))
            # update global frontier as if this msg is 'seen'
            for q, c in vc.items():
                if c > global_seen.get(q, 0):
                    global_seen[q] = c
            mid += 1
    # arrival order
    order = list(range(len(msgs)))
    if arrival == "reverse":
        # reverse within each sender: send high counts first so they buffer
        by_sender = {}
        for i, (m2, p2, v2) in enumerate(msgs):
            by_sender.setdefault(p2, []).append(i)
        order = []
        maxlen = max((len(v) for v in by_sender.values()), default=0)
        for j in range(maxlen):
            for p in range(n_senders):
                lst = by_sender.get(p, [])
                if j < len(lst):
                    order.append(lst[len(lst) - 1 - j])  # reverse per sender
    elif arrival == "shuffle":
        rnd.shuffle(order)
    elif arrival == "inorder":
        pass
    return [msgs[i] for i in order], msgs


def _scenarios():
    rnd = random.Random(20260726)
    scen = []  # (name, arrival_list)
    scen.append(("single_inorder", [(0, 0, {0: 1}), (1, 0, {0: 2}), (2, 0, {0: 3})]))
    scen.append(("single_reversed", [(2, 0, {0: 3}), (1, 0, {0: 2}), (0, 0, {0: 1})]))
    scen.append(("two_senders_indep",
                 [(0, 0, {0: 1}), (1, 1, {1: 1}), (2, 0, {0: 2}), (3, 1, {1: 2})]))
    scen.append(("cross_dep_ordered",
                 [(0, 0, {0: 1}), (1, 1, {0: 1, 1: 1}), (2, 0, {0: 2})]))
    scen.append(("cross_dep_blocked",
                 [(1, 1, {0: 1, 1: 1}), (0, 0, {0: 1}), (2, 0, {0: 2})]))
    scen.append(("concurrent",
                 [(0, 0, {0: 1}), (1, 1, {1: 1})]))
    for c in range(20):
        ns = rnd.choice([1, 2, 3, 4])
        ps = rnd.choice([1, 3, 6, 12])
        arr = rnd.choice(["reverse", "shuffle", "inorder"])
        alist, _ = _gen_causal_stream(random.Random(1000 + c), ns, ps, arr)
        scen.append(("rand%d_%s_s%d_p%d" % (c, arr, ns, ps), alist))
    return scen


def run_correctness():
    m = scope_pkg()
    npass = 0
    results = {}
    for name, alist in _scenarios():
        try:
            ch = m.CausalChannel()
            ref = _RefChannel()
            ok = True
            for msg in alist:
                got = ch.deliver(msg)
                exp = ref.deliver(msg)
                if got != exp:
                    ok = False
                if ch.pending_count() != len(ref.pending):
                    ok = False
            if ch.delivered_ids() != ref.delivered:
                ok = False
            if ch.clock() != ref.clock():
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
    ns = int(os.environ.get("WRO_CAUSAL_SENDERS", "4"))
    per = int(os.environ.get("WRO_CAUSAL_PER", "1500"))
    rounds = int(os.environ.get("WRO_CAUSAL_ROUNDS", "3"))
    alist, _ = _gen_causal_stream(random.Random(99), ns, per, arrival="reverse")

    def one():
        ch = m.CausalChannel()
        total = 0
        for msg in alist:
            total += len(ch.deliver(msg))
        return total

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
            print("WRO_CAUSAL_RESULT " + json.dumps({"timing_ms": run_timing()}))
        except Exception as e:
            import traceback
            print("WRO_CAUSAL_RESULT " + json.dumps({"timing_ms": -1, "error": repr(e),
                                                    "tb": traceback.format_exc()[-800:]}))
        return
    origin = None
    try:
        origin = os.path.realpath(scope_pkg().channel.__file__)
    except Exception:
        pass
    try:
        frac, total, failed = run_correctness()
        print("WRO_CAUSAL_RESULT " + json.dumps(
            {"correctness_frac": frac, "n_cases": total, "n_failed": len(failed),
             "failed": {k: failed[k] for k in list(failed)[:8]}, "origin": origin}))
    except Exception as e:
        import traceback
        print("WRO_CAUSAL_RESULT " + json.dumps(
            {"correctness_frac": 0.0, "error": repr(e),
             "tb": traceback.format_exc()[-900:], "origin": origin}))


if __name__ == "__main__":
    main()
