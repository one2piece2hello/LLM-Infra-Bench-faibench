"""Tensor liveness model shared by the memory-accounting simulator.

Fixed building block (out of the editable scope): the data model for a training/inference execution
plan. A ``Tensor`` is allocated at one step and freed at another (half-open ``[alloc, free)`` in
step index) and occupies ``nbytes`` bytes while live. An ``ExecutionPlan`` is an ordered list of
such tensors plus the total number of steps. All peak-memory results are defined in terms of this
model: a tensor contributes ``nbytes`` to every step ``s`` with ``alloc <= s < free``.
"""
from __future__ import annotations


class Tensor:
    __slots__ = ("name", "nbytes", "alloc", "free")

    def __init__(self, name, nbytes, alloc, free):
        if nbytes < 0:
            raise ValueError("nbytes < 0")
        if free < alloc:
            raise ValueError("free < alloc")
        self.name = name
        self.nbytes = int(nbytes)
        self.alloc = int(alloc)
        self.free = int(free)

    def live_at(self, step):
        return self.alloc <= step < self.free

    def as_tuple(self):
        return (self.name, self.nbytes, self.alloc, self.free)

    def __repr__(self):
        return "Tensor(%r, %d, %d, %d)" % (self.name, self.nbytes, self.alloc, self.free)


class ExecutionPlan:
    def __init__(self, n_steps, tensors=None):
        self.n_steps = int(n_steps)
        self.tensors = list(tensors) if tensors else []

    def add(self, tensor):
        self.tensors.append(tensor)
