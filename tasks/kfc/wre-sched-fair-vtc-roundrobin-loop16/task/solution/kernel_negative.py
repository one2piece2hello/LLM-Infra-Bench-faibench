# NEGATIVE variant (reviewer-only; never baked into the image). FAST-but-WRONG.
# Breaks the round-robin fairness invariant: it orders by TENANT FIRST (all of tenant 0's requests,
# then all of tenant 1's, ...) instead of by round first. This is exactly the starvation the fair
# scheduler is meant to prevent — a bursty tenant monopolises the head of the schedule. The contract
# mandates ordering by (round asc, tenant asc); ordering by (tenant asc, arrival) disagrees with the
# reference on every input that has more than one request per tenant. Vectorized and fast, but
# correctness FAILS.
import numpy as np


def fair_interleave_order(tenant_ids, num_tenants):
    t = np.asarray(tenant_ids, dtype=np.int64)
    n = int(t.shape[0])
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
    order = np.argsort(t, kind="stable")          # BUG: tenant-first grouping, not round-first
    out[order] = np.arange(n, dtype=np.int64)
    return out


def custom_kernel(data):
    tenant_ids, num_tenants = data
    return fair_interleave_order(tenant_ids, num_tenants)
