"""Vector clock primitive shared by the causal-delivery subsystem.

Fixed building block (out of the editable scope): a vector clock over a fixed set of process ids.
All causal-ordering decisions in the subsystem are defined in terms of the partial order this module
implements.

A vector clock is a mapping ``pid -> int`` (missing entries are 0). For clocks ``a`` and ``b``:
  * ``a <= b`` iff ``a[p] <= b[p]`` for every p (dominates).
  * ``a < b`` iff ``a <= b`` and ``a != b`` (strictly precedes / happens-before).
  * ``a`` and ``b`` are concurrent iff neither ``a <= b`` nor ``b <= a``.
"""
from __future__ import annotations


class VectorClock:
    def __init__(self, counts=None):
        self.counts = dict(counts) if counts else {}

    def get(self, pid):
        return self.counts.get(pid, 0)

    def copy(self):
        return VectorClock(self.counts)

    def tick(self, pid):
        self.counts[pid] = self.counts.get(pid, 0) + 1
        return self

    def merge(self, other):
        for p, v in other.counts.items():
            if v > self.counts.get(p, 0):
                self.counts[p] = v
        return self

    def leq(self, other):
        for p, v in self.counts.items():
            if v > other.counts.get(p, 0):
                return False
        return True

    def lt(self, other):
        return self.leq(other) and not self._eq(other)

    def _eq(self, other):
        keys = set(self.counts) | set(other.counts)
        return all(self.counts.get(k, 0) == other.counts.get(k, 0) for k in keys)

    def concurrent(self, other):
        return not self.leq(other) and not other.leq(self)

    def as_dict(self):
        return {p: v for p, v in self.counts.items() if v != 0}
