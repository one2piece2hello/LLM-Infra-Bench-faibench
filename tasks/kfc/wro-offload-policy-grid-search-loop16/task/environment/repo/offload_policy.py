"""Heterogeneous-offload policy search for single-GPU LLM inference.

A large model does not fit in GPU memory, so every tensor family is *tiered*
across GPU / CPU / NVMe. A policy is six fractions:

    wg, wc   fraction of the WEIGHTS  kept on GPU / in CPU DRAM (rest on disk)
    cg, cc   fraction of the KV CACHE kept on GPU / in CPU DRAM (rest on disk)
    hg, hc   fraction of the ACTIVATIONS (hidden states) on GPU / CPU

with ``0 <= g``, ``0 <= c``, ``g + c <= 1`` per family. Choosing a policy is a
constrained minimisation: the analytic cost model says how slow a policy is, the
memory model says whether it even fits, and the search enumerates a simplex grid
and takes the best feasible point.

``specs`` is a dict of hardware/model constants:
    n_layers, flops_per_layer, gpu_tflops,
    weight_bytes_per_layer, kv_bytes_per_layer, act_bytes_per_layer,
    pcie_bytes_per_s, disk_bytes_per_s, gpu_working_set_bytes, pinned_buf_bytes,
    prefill_overhead_s

``budgets`` is ``(gpu_budget_bytes, cpu_budget_bytes)``.

Everything here is float64 elementwise arithmetic over the policy grid. The cost
model deliberately uses a FIXED sequence of elementwise operations, and the
contract is *exact float equality*, so the operation order and the association
must be preserved (do not re-associate a sum, do not turn ``a * b / c`` into
``a * (b / c)``).
"""

import numpy as np

def enumerate_grid(steps):
    """Enumerate the policy grid as an ``(N, 6)`` float64 array.

    For one tensor family the admissible ``(g, c)`` pairs on a ``steps``-point
    grid are, in this exact order::

        for gi in range(steps + 1):
            for ci in range(steps + 1 - gi):
                (gi / steps, ci / steps)

    so ``P = (steps + 1) * (steps + 2) // 2`` pairs. The full grid is the
    lexicographic product over (weights, kv-cache, activations)::

        for pw in range(P):
            for pk in range(P):
                for pa in range(P):
                    row = (wg, wc, cg, cc, hg, hc)

    giving ``N = P ** 3`` rows. Column order is exactly
    ``(wg, wc, cg, cc, hg, hc)``.
    """
    # SLOW-BUT-CORRECT reference path: python triple product, row by row.
    steps = int(steps)
    pairs = []
    for gi in range(steps + 1):
        for ci in range(steps + 1 - gi):
            pairs.append((gi / steps, ci / steps))
    rows = []
    for pw in pairs:
        for pk in pairs:
            for pa in pairs:
                rows.append((pw[0], pw[1], pk[0], pk[1], pa[0], pa[1]))
    return np.array(rows, dtype=np.float64)

def policy_peak_memory(specs, pol):
    """Peak GPU and CPU bytes held by each policy row.

    ``pol`` is an ``(N, 6)`` array (or a single length-6 row). Returns
    ``(gpu_bytes, cpu_bytes)``, each shaped like ``pol[:, 0]``::

        gpu = n_layers * (wg * W + cg * C + hg * A) + gpu_working_set_bytes
        cpu = n_layers * (wc * W + cc * C + hc * A) + pinned_buf_bytes

    with ``W = weight_bytes_per_layer``, ``C = kv_bytes_per_layer``,
    ``A = act_bytes_per_layer``. The multiplication order shown is part of the
    contract.
    """
    p = np.atleast_2d(np.asarray(pol, dtype=np.float64))
    n = int(specs["n_layers"])
    W = float(specs["weight_bytes_per_layer"])
    C = float(specs["kv_bytes_per_layer"])
    A = float(specs["act_bytes_per_layer"])
    ws = float(specs["gpu_working_set_bytes"])
    pb = float(specs["pinned_buf_bytes"])
    gpu = np.empty(p.shape[0], dtype=np.float64)
    cpu = np.empty(p.shape[0], dtype=np.float64)
    for i in range(p.shape[0]):
        wg = p[i, 0]; wc = p[i, 1]; cg = p[i, 2]
        cc = p[i, 3]; hg = p[i, 4]; hc = p[i, 5]
        gpu[i] = n * (wg * W + cg * C + hg * A) + ws
        cpu[i] = n * (wc * W + cc * C + hc * A) + pb
    return gpu, cpu

def policy_latency(specs, pol):
    """Estimated end-to-end latency in seconds for each policy row.

    Per decoder layer the GPU compute time is
    ``t_c = flops_per_layer / (gpu_tflops * 1e12)``, and the bytes that must be
    *fetched* are the non-resident fractions, moved over PCIe from CPU DRAM and
    over the NVMe link from disk::

        w_disk = 1.0 - wg - wc
        t_w    = wc * W / pcie + w_disk * W / disk
        k_disk = 1.0 - cg - cc
        t_k    = cc * C / pcie + k_disk * C / disk
        a_disk = 1.0 - hg - hc
        t_a    = hc * A / pcie + a_disk * A / disk

    Transfers overlap with compute, so the layer takes
    ``t_layer = max(t_c, t_w + t_k + t_a)`` and the model takes
    ``n_layers * t_layer + prefill_overhead_s``.

    Returns a float64 array shaped like ``pol[:, 0]``. Keep the exact expression
    order above: the contract is bit-exact float equality.
    """
    p = np.atleast_2d(np.asarray(pol, dtype=np.float64))
    n = int(specs["n_layers"])
    W = float(specs["weight_bytes_per_layer"])
    C = float(specs["kv_bytes_per_layer"])
    A = float(specs["act_bytes_per_layer"])
    pcie = float(specs["pcie_bytes_per_s"])
    disk = float(specs["disk_bytes_per_s"])
    t_c = float(specs["flops_per_layer"]) / (float(specs["gpu_tflops"]) * 1e12)
    over = float(specs["prefill_overhead_s"])
    out = np.empty(p.shape[0], dtype=np.float64)
    for i in range(p.shape[0]):
        wg = p[i, 0]; wc = p[i, 1]; cg = p[i, 2]
        cc = p[i, 3]; hg = p[i, 4]; hc = p[i, 5]
        w_disk = 1.0 - wg - wc
        t_w = wc * W / pcie + w_disk * W / disk
        k_disk = 1.0 - cg - cc
        t_k = cc * C / pcie + k_disk * C / disk
        a_disk = 1.0 - hg - hc
        t_a = hc * A / pcie + a_disk * A / disk
        s = t_w + t_k + t_a
        t_layer = t_c if t_c > s else s
        out[i] = n * t_layer + over
    return out

def feasible_mask(specs, pol, budgets):
    """Boolean mask: which policy rows fit both memory budgets.

    ``budgets`` is ``(gpu_budget_bytes, cpu_budget_bytes)``; a row is feasible
    iff ``gpu_bytes <= gpu_budget`` AND ``cpu_bytes <= cpu_budget`` using
    :func:`policy_peak_memory`. Returns a bool array shaped like ``pol[:, 0]``.
    """
    gpu, cpu = policy_peak_memory(specs, pol)
    gb = float(budgets[0])
    cb = float(budgets[1])
    out = np.empty(gpu.shape[0], dtype=bool)
    for i in range(gpu.shape[0]):
        out[i] = bool(gpu[i] <= gb and cpu[i] <= cb)
    return out

def search_best_policy(specs, budgets, steps):
    """Enumerate, filter and pick the fastest feasible policy.

    Ties are broken by **first in enumeration order** (a strict ``<``
    improvement test). Returns a dict::

        {"best_index":   int index into the enumerated grid, -1 if none feasible,
         "best_policy":  list of 6 floats (empty list if none feasible),
         "best_latency": float seconds (-1.0 if none feasible),
         "gpu_bytes":    float peak GPU bytes of the winner (-1.0 if none),
         "cpu_bytes":    float peak CPU bytes of the winner (-1.0 if none),
         "n_feasible":   int, "n_candidates": int}
    """
    grid = enumerate_grid(steps)
    ok = feasible_mask(specs, grid, budgets)
    lat = policy_latency(specs, grid)
    gpu, cpu = policy_peak_memory(specs, grid)
    best = -1
    best_lat = 0.0
    n_feas = 0
    for i in range(grid.shape[0]):
        if not ok[i]:
            continue
        n_feas += 1
        if best < 0 or lat[i] < best_lat:
            best = i
            best_lat = lat[i]
    if best < 0:
        return {"best_index": -1, "best_policy": [], "best_latency": -1.0,
                "gpu_bytes": -1.0, "cpu_bytes": -1.0, "n_feasible": 0,
                "n_candidates": int(grid.shape[0])}
    return {"best_index": int(best),
            "best_policy": [float(x) for x in grid[best]],
            "best_latency": float(best_lat),
            "gpu_bytes": float(gpu[best]), "cpu_bytes": float(cpu[best]),
            "n_feasible": int(n_feas), "n_candidates": int(grid.shape[0])}

