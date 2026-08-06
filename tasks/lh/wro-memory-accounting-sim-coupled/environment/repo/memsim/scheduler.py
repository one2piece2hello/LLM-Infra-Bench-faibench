"""Rematerialization / eviction scheduler over an execution plan (perf-modeling backend).

Given an ``ExecutionPlan`` and a memory budget, the scheduler decides which tensors to evict-early
(shorten liveness) to fit the budget, by asking the ``MemoryAccountant`` for the resulting peak.

Observable contract:
  * ``fits(budget)`` -> True iff the plan's current peak() bytes <= budget.
  * ``over_budget_steps(budget)`` -> sorted list of steps whose footprint exceeds ``budget``.
  * ``best_eviction(budget)`` -> among tensors live at the CURRENT peak step, the name whose
    ``peak_after_free`` gives the smallest resulting peak (ties broken by name); ``None`` if the
    plan already fits or no tensor is live at the peak step.
  * ``greedy_plan(budget, max_evictions)`` -> apply up to ``max_evictions`` rounds of best_eviction
    (each shrinks the chosen tensor's free by 1, clamped), returning the list of evicted names in
    order; stops early once ``fits(budget)`` or no candidate remains.
"""
from __future__ import annotations

from .accountant import MemoryAccountant
from .model import ExecutionPlan


class EvictionScheduler:
    def __init__(self, plan: ExecutionPlan):
        self.plan = plan
        self.acct = MemoryAccountant(plan)

    def fits(self, budget):
        return self.acct.peak()[0] <= budget

    def over_budget_steps(self, budget):
        tl = self.acct.timeline()
        return [s for s, v in enumerate(tl) if v > budget]

    def best_eviction(self, budget):
        peak_bytes, peak_step = self.acct.peak()
        if peak_bytes <= budget:
            return None
        # candidates = tensors live at the peak step
        cands = [t.name for t in self.plan.tensors if t.alloc <= peak_step < t.free]
        if not cands:
            return None
        best_name = None
        best_peak = None
        for name in sorted(set(cands)):
            pk = self.acct.peak_after_free(name)
            if best_peak is None or pk < best_peak:
                best_peak = pk
                best_name = name
        return best_name

    def greedy_plan(self, budget, max_evictions):
        evicted = []
        for _ in range(int(max_evictions)):
            if self.fits(budget):
                break
            name = self.best_eviction(budget)
            if name is None:
                break
            for t in self.plan.tensors:
                if t.name == name:
                    t.free = max(t.alloc, t.free - 1)
                    break
            evicted.append(name)
        return evicted
