# Performance Optimization Task — submission entry point.
#
# Implement `fair_interleave_order` to the contract in instruction.md, then make it as fast as
# possible. This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import numpy as np  # noqa: F401  (available; you may use it)


def fair_interleave_order(tenant_ids, num_tenants):
    """Per-tenant fair round-robin interleave order for a multi-tenant serving queue.

    A fairness-aware serving scheduler (VTC-style: no tenant should be starved by a bursty
    neighbour) does not serve a waiting queue in pure arrival order — instead it interleaves
    tenants round-robin. Conceptually: take each tenant's requests in arrival order; serve every
    tenant's 1st request (in ascending tenant id), then every tenant's 2nd request, and so on.
    A tenant that submitted a burst of requests does not monopolise the head of the schedule.

    Contract (deterministic; all correct implementations agree exactly):
      * ``tenant_ids``: 1-D sequence of ``N`` integers in ``[0, num_tenants)`` — the owning tenant of
        each waiting request, given in **arrival order** (index = arrival position).
      * ``num_tenants``: a positive ``int`` — the number of tenants (ids ``0 .. num_tenants-1``).
      * For each request compute its **within-tenant round** ``r`` = how many EARLIER requests
        (smaller original index) share its tenant (so a tenant's requests get rounds ``0, 1, 2, …``
        in arrival order).
      * The fair schedule orders all requests by ``(round asc, tenant_id asc)``; within one
        ``(round, tenant)`` there is at most one request, so this is a total order.
      * Return a 1-D ``numpy`` ``int64`` array ``sched_pos`` of length ``N``: entry ``i`` is the
        **0-based position** of request ``i`` in that fair round-robin schedule.

    Args:
        tenant_ids:  1-D sequence of N integer tenant ids in [0, num_tenants), arrival order.
        num_tenants: positive int, number of tenants.

    Return:
        numpy.ndarray[int64] of shape (N,): the fair-schedule position of each request.
    """
    raise NotImplementedError("implement fair_interleave_order to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls. data = (tenant_ids, num_tenants). Already wired — returns
    the per-request fair round-robin schedule-position array from fair_interleave_order."""
    tenant_ids, num_tenants = data
    return fair_interleave_order(tenant_ids, num_tenants)
