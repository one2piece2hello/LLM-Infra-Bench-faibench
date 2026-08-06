# ORACLE variant (reviewer-only; never baked into the image). FAST host-logic implementation.
# The routing is INHERENTLY SEQUENTIAL (each choice depends on the loads produced by all prior
# requests), so there is no vectorization. The gradient is pure per-request interpreter overhead:
# convert the inputs to python lists ONCE up front, run the power-of-two-choices loop entirely on
# python ints (plain list indexing + int compares/increments, no per-element numpy scalar boxing),
# then materialize the two result arrays ONCE at the end. Same result as the naive numpy-scalar
# loop, but each request costs a handful of cheap python-list ops instead of ~10-30x-slower numpy
# 0-d scalar get/set.
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
        if load[a] <= load[b]:
            c = a
        else:
            c = b
        choices[i] = c
        load[c] += 1
    return np.array(choices, dtype=np.int64), np.array(load, dtype=np.int64)


def custom_kernel(data):
    num_replicas, cand_a, cand_b, init_load = data
    return route_p2c(num_replicas, cand_a, cand_b, init_load)
