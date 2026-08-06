"""Peak-memory accountant over an execution plan (perf-modeling / simulation backend).

Given an ``ExecutionPlan`` (tensors with ``[alloc, free)`` liveness in step index), the simulator
reports the memory footprint per step and the peak, as a memory planner / OOM predictor does.

Observable contract (bytes are ints; steps are ``0..plan.n_steps-1``):
  * ``footprint_at(step)`` -> total live bytes at ``step`` (sum of nbytes of tensors live there).
  * ``timeline()`` -> list of length ``plan.n_steps`` giving footprint at each step.
  * ``peak()`` -> (peak_bytes, peak_step) where peak_step is the SMALLEST step achieving the max;
    (0, 0) for an empty plan or zero steps.
  * ``peak_after_free(name)`` -> the peak the plan WOULD have if the tensor ``name`` were freed one
    step earlier (free -= 1, clamped so free >= alloc); used to score eviction candidates. Returns
    the recomputed peak_bytes. Unknown name -> current peak_bytes.
"""
from __future__ import annotations

from .model import ExecutionPlan


class MemoryAccountant:
    def __init__(self, plan: ExecutionPlan):
        self.plan = plan

    def footprint_at(self, step):
        total = 0
        for t in self.plan.tensors:
            if t.alloc <= step < t.free:
                total += t.nbytes
        return total

    def timeline(self):
        return [self.footprint_at(s) for s in range(self.plan.n_steps)]

    def peak(self):
        tl = self.timeline()
        if not tl:
            return (0, 0)
        best = tl[0]
        best_step = 0
        for s in range(1, len(tl)):
            if tl[s] > best:
                best = tl[s]
                best_step = s
        return (best, best_step)

    def peak_after_free(self, name):
        tensors = self.plan.tensors
        target = None
        for t in tensors:
            if t.name == name:
                target = t
                break
        if target is None:
            return self.peak()[0]
        old_free = target.free
        new_free = max(target.alloc, old_free - 1)
        target.free = new_free
        try:
            pk = self.peak()[0]
        finally:
            target.free = old_free
        return pk
