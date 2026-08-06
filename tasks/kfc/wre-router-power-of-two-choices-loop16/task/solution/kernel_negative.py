# NEGATIVE variant (reviewer-only; never baked into the image). FAST-but-WRONG.
# It is just as fast as the oracle (same python-list loop), but it breaks the core power-of-two-
# choices invariant: it routes each request to the MORE-loaded of the two sampled replicas
# (>= instead of <=). Ties still go to `a` (so tie behaviour matches), but every strictly-unequal
# request is routed to the wrong replica, and the divergence compounds as the load array evolves,
# so the returned choices/final_load do not match the contract -> correctness FAILS.
import numpy as np


def route_p2c(num_replicas, cand_a, cand_b, init_load):
    ca = np.asarray(cand_a).tolist()
    cb = np.asarray(cand_b).tolist()
    load = np.asarray(init_load, dtype=np.int64).tolist()
    n = len(ca)
    choices = [0] * n
    for i in range(n):
        a = ca[i]
        b = cb[i]
        if load[a] >= load[b]:   # BUG: picks the MORE-loaded replica (should be <=)
            c = a
        else:
            c = b
        choices[i] = c
        load[c] += 1
    return np.array(choices, dtype=np.int64), np.array(load, dtype=np.int64)


def custom_kernel(data):
    num_replicas, cand_a, cand_b, init_load = data
    return route_p2c(num_replicas, cand_a, cand_b, init_load)
