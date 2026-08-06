"""Empirical Big-O complexity estimation for a microbenchmark harness.

When a benchmark is run across a sweep of workload sizes, a harness can infer the
empirical asymptotic complexity of the measured code by least-squares fitting the
observed runtimes against a family of candidate complexity curves and reporting the
best-fitting one together with its leading coefficient. This mirrors Google
Benchmark's ``ComputeBigO`` / ``MinimalLeastSq`` (``src/complexity.cc``).

``compute_bigo`` fits, for every benchmark, the ``B`` rows of a ``(B, K)`` runtime
table against the ``K`` workload sizes ``ns`` and returns the inferred complexity.

Contract (exactly Google Benchmark's definitions):
    Candidate curves ``g(n)`` in fixed order
    ``[ "(1)"=1 , "lgN"=log2(n) , "N"=n , "NlgN"=n*log2(n) , "N^2"=n^2 , "N^3"=n^3 ]``.
    For each benchmark row ``t`` (length K) and each curve ``g``:
        * ``coef = sum_i( t_i * g_i ) / sum_i( g_i^2 )``     (least squares through origin)
        * ``rms  = sqrt( sum_i( (t_i - coef*g_i)^2 ) / K ) / mean(t)``   (mean-normalized RMS)
    The chosen complexity is the curve with the SMALLEST ``rms``; ties resolve to the
    EARLIER curve in the fixed order above (i.e. "(1)" is the default and is only
    displaced by a strictly smaller rms), matching ``ComputeBigO``'s ``o1``-default +
    strict-improvement scan over ``{lgN, N, NlgN, N^2, N^3}``.

Returns a ``dict`` with:
    * ``"complexity"`` : list of ``B`` strings (the chosen curve label per benchmark);
    * ``"coef"``       : ``float64`` array ``(B,)`` -- the chosen curve's coefficient;
    * ``"rms"``        : ``float64`` array ``(B,)`` -- the chosen curve's mean-normalized RMS.
"""
# NAIVE_PER_CURVE_PYTHON_FIT: correct but slow reference fit -- an explicit
# double loop over benchmarks x candidate curves with scalar Python accumulators
# for the least-squares sums and the residual RMS. Behaviour-equivalent to the
# vectorized batch fit; just slow.
import math

import numpy as np

_LABELS = ["(1)", "lgN", "N", "NlgN", "N^2", "N^3"]


def _curve(kind, n):
    if kind == 0:
        return 1.0
    if kind == 1:
        return math.log2(n)
    if kind == 2:
        return float(n)
    if kind == 3:
        return float(n) * math.log2(n)
    if kind == 4:
        return float(n) * float(n)
    return float(n) * float(n) * float(n)


def compute_bigo(ns, times):
    """Infer per-benchmark empirical complexity by least-squares curve fitting.

    :param ns: array-like of shape ``(K,)`` -- the workload sizes (each ``>= 1``).
    :param times: array-like of shape ``(B, K)`` -- measured runtimes; ``times[b]``
        are benchmark ``b``'s runtimes at each size in ``ns``. Cast to ``float64``;
        inputs are not mutated.
    :returns: ``dict`` with keys ``"complexity"`` (list of ``B`` label strings),
        ``"coef"`` and ``"rms"`` (``float64`` arrays of shape ``(B,)``), per the
        module docstring.
    """
    ns = np.asarray(ns, dtype=np.float64).ravel()
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != ns.shape[0]:
        raise ValueError("times must have shape (B, K) matching ns of shape (K,)")
    B, K = t.shape
    complexity = []
    coef_out = np.empty(B, dtype=np.float64)
    rms_out = np.empty(B, dtype=np.float64)
    for b in range(B):
        best_j = 0
        best_coef = 0.0
        best_rms = None
        mean_t = 0.0
        for i in range(K):
            mean_t += float(t[b, i])
        mean_t /= K
        for j in range(6):
            sigma_g2 = 0.0
            sigma_tg = 0.0
            for i in range(K):
                g = _curve(j, ns[i])
                sigma_g2 += g * g
                sigma_tg += float(t[b, i]) * g
            coef = sigma_tg / sigma_g2
            resid = 0.0
            for i in range(K):
                fit = coef * _curve(j, ns[i])
                d = float(t[b, i]) - fit
                resid += d * d
            rms = math.sqrt(resid / K) / mean_t
            if best_rms is None or rms < best_rms:
                best_rms = rms
                best_coef = coef
                best_j = j
        complexity.append(_LABELS[best_j])
        coef_out[b] = best_coef
        rms_out[b] = best_rms
    return {"complexity": complexity, "coef": coef_out, "rms": rms_out}
