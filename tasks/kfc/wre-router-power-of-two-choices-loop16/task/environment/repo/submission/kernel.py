# Performance Optimization Task — submission entry point.
#
# Implement `route_p2c` to the contract in instruction.md, then make it as fast as possible.
# This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import numpy as np  # noqa: F401  (available; you may use it)


def route_p2c(num_replicas, cand_a, cand_b, init_load):
    """Route a stream of requests across replicas by the "power of two choices" rule.

    A load-balancing request router places each incoming request on one of ``num_replicas``
    backend replicas. For every request it is handed two candidate replica indices (sampled
    upstream) and must send the request to whichever of the two is currently less loaded, then
    account for the new request on that replica. Routing is done for a whole batch of ``N``
    sequential requests, starting from a known per-replica load.

    Contract (deterministic; all correct implementations agree exactly):
      * ``num_replicas``: an ``int`` ``R`` (the number of replicas).
      * ``cand_a``, ``cand_b``: 1-D integer arrays of length ``N`` — for request ``i`` the two
        candidate replica indices are ``cand_a[i]`` and ``cand_b[i]`` (each in ``[0, R)``).
      * ``init_load``: a 1-D integer array of length ``R`` — the starting load of each replica.
      * Maintain a mutable ``load`` array initialised to a copy of ``init_load``. Process the
        requests IN ORDER. For request ``i`` let ``a = cand_a[i]``, ``b = cand_b[i]`` and choose
        ``chosen = a if load[a] <= load[b] else b`` (a TIE goes to ``a``). Record ``chosen`` as the
        source for request ``i``, then increment ``load[chosen]`` by 1 before the next request.
      * Return a tuple ``(choices, final_load)`` where ``choices`` is a 1-D ``numpy`` ``int64``
        array of length ``N`` (the chosen replica per request, in order) and ``final_load`` is a
        1-D ``numpy`` ``int64`` array of length ``R`` (the load array after all requests).

    Args:
        num_replicas: int R, the number of replicas.
        cand_a: 1-D int array of length N, first candidate replica per request.
        cand_b: 1-D int array of length N, second candidate replica per request.
        init_load: 1-D int array of length R, starting load per replica.

    Return:
        (numpy.ndarray[int64] shape (N,), numpy.ndarray[int64] shape (R,)):
        the per-request chosen replica and the final per-replica load.
    """
    raise NotImplementedError("implement route_p2c to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls. data = (num_replicas, cand_a, cand_b, init_load). Already
    wired — returns the (choices, final_load) tuple from route_p2c."""
    num_replicas, cand_a, cand_b, init_load = data
    return route_p2c(num_replicas, cand_a, cand_b, init_load)
