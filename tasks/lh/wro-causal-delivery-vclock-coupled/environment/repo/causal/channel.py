"""Causal-delivery channel / driver (distributed-correctness backend).

Drives causal-ordered delivery for one receiver: each arriving message is buffered, then the driver
delivers every message that is (transitively) now deliverable, in a deterministic order, updating the
delivered vector clock.

Observable contract:
  * ``deliver(msg)`` where ``msg`` = ``(msg_id, sender_pid, vc_dict)`` -> list of msg_ids delivered as
    a result of this arrival, in delivery order (empty if buffered; possibly several if it unblocks a
    chain). Among simultaneously deliverable messages, the smallest ``msg_id`` goes first.
  * ``delivered_ids()`` -> the full delivery order so far.
  * ``pending_count()`` -> number of buffered messages.
  * ``clock()`` -> current delivered-clock as a dict (nonzero entries).
"""
from __future__ import annotations

from .buffer import PendingStore, is_deliverable
from .vclock import VectorClock


class CausalChannel:
    def __init__(self):
        self.D = VectorClock()
        self.store = PendingStore()
        self._delivered = []

    def _next_deliverable(self):
        best = None
        for (mid, pid, vc) in self.store.all():
            if is_deliverable(pid, vc, self.D):
                if best is None or mid < best[0]:
                    best = (mid, pid, vc)
        return best

    def deliver(self, msg):
        msg_id, pid, vc_dict = msg
        self.store.add(msg_id, pid, VectorClock(vc_dict))
        out = []
        while True:
            nxt = self._next_deliverable()
            if nxt is None:
                break
            mid, p, vc = nxt
            self.D.tick(p)
            self.store.remove(mid)
            self._delivered.append(mid)
            out.append(mid)
        return out

    def delivered_ids(self):
        return list(self._delivered)

    def pending_count(self):
        return self.store.count()

    def clock(self):
        return self.D.as_dict()
