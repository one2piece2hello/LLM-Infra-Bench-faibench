"""Pending-message store for the causal-delivery subsystem (distributed-correctness backend).

Buffers messages that have arrived but are not yet causally deliverable, and exposes the buffered
messages to the delivery driver in ``channel.py``.

A message is ``(msg_id, sender_pid, vc)`` where ``vc`` is a VectorClock (the sender's clock after its
own tick, i.e. ``vc[sender] = D_sender[sender] + 1``). Deliverability against a delivered-clock ``D``:
    vc[sender] == D[sender] + 1   and   vc[k] <= D[k] for every k != sender.
"""
from __future__ import annotations

from .vclock import VectorClock


def is_deliverable(pid, vc, D):
    if vc.get(pid) != D.get(pid) + 1:
        return False
    for k, v in vc.as_dict().items():
        if k == pid:
            continue
        if v > D.get(k):
            return False
    return True


class PendingStore:
    def __init__(self):
        self._pending = []          # list of (msg_id, pid, vc)

    def add(self, msg_id, pid, vc):
        self._pending.append((msg_id, pid, vc))

    def remove(self, msg_id):
        for i, (mid, _, _) in enumerate(self._pending):
            if mid == msg_id:
                self._pending.pop(i)
                return

    def count(self):
        return len(self._pending)

    def all(self):
        return list(self._pending)
