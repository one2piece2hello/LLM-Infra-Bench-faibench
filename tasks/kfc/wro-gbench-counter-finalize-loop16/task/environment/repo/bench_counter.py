"""Per-counter finalization for a microbenchmark harness.

A microbenchmark records, for every benchmark case, a set of *user counters*.
Each counter carries a raw accumulated ``value`` plus a bitmask of ``flags`` that
describes how the raw value must be finalized once the run's ``iterations``,
``cpu_time`` (seconds) and ``num_threads`` are known. This mirrors Google
Benchmark's ``Finish`` (``src/counter.cc``) together with the flag semantics
declared in ``include/benchmark/counter.h``.

``finalize_counters`` takes the whole counter table for a run -- ``B`` benchmarks
each with ``C`` counters -- and returns the finalized value of every counter.

Contract (exactly Google Benchmark's ``Finish`` order; the transforms are applied
in this fixed sequence to each counter's raw value ``v``):

    * ``kIsRate``              (``1 << 0``): ``v /= cpu_time``
    * ``kAvgThreads``          (``1 << 1``): ``v /= num_threads``
    * ``kIsIterationInvariant``(``1 << 2``): ``v *= iterations``
    * ``kAvgIterations``       (``1 << 3``): ``v /= iterations``
    * ``kInvert``              (``1 << 31``): ``v = 1.0 / v``  -- applied *last*, always.

``cpu_time``, ``num_threads`` and ``iterations`` are supplied per benchmark (one
value per row) and broadcast across that benchmark's counters. The result is a
``float64`` array of shape ``(B, C)``.
"""
# DEGRADED_BASELINE_SCALAR_LOOP: correct but slow reference finalization -- an
# explicit per-(benchmark, counter) Python loop that reads each raw value out of
# the numpy table one cell at a time and applies the five flag transforms with
# scalar Python branches. Behaviour-equivalent to the vectorized form; just slow.
import numpy as np

kIsRate = 1 << 0
kAvgThreads = 1 << 1
kIsIterationInvariant = 1 << 2
kAvgIterations = 1 << 3
kInvert = 1 << 31


def finalize_counters(values, flags, iterations, cpu_time, num_threads):
    """Finalize a ``(B, C)`` counter table according to per-counter flags.

    :param values: array-like ``(B, C)`` -- raw accumulated counter values.
    :param flags: array-like ``(B, C)`` of integer flag bitmasks.
    :param iterations: array-like ``(B,)`` -- iteration count per benchmark.
    :param cpu_time: array-like ``(B,)`` -- cpu time (seconds) per benchmark.
    :param num_threads: array-like ``(B,)`` -- thread count per benchmark.
    :returns: ``float64`` ``numpy.ndarray`` of shape ``(B, C)`` with each counter
        finalized per the flag order in the module docstring. Inputs are not
        mutated.
    """
    vals = np.asarray(values, dtype=np.float64)
    fl = np.asarray(flags, dtype=np.int64)
    if vals.ndim != 2 or fl.shape != vals.shape:
        raise ValueError("values and flags must both have shape (B, C)")
    B, C = vals.shape
    it = np.asarray(iterations, dtype=np.float64)
    ct = np.asarray(cpu_time, dtype=np.float64)
    nt = np.asarray(num_threads, dtype=np.float64)
    out = np.empty((B, C), dtype=np.float64)
    for b in range(B):
        it_b = float(it[b])
        ct_b = float(ct[b])
        nt_b = float(nt[b])
        for c in range(C):
            v = float(vals[b, c])
            f = int(fl[b, c])
            if f & kIsRate:
                v = v / ct_b
            if f & kAvgThreads:
                v = v / nt_b
            if f & kIsIterationInvariant:
                v = v * it_b
            if f & kAvgIterations:
                v = v / it_b
            if f & kInvert:
                v = 1.0 / v
            out[b, c] = v
    return out
