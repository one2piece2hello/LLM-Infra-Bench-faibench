#!/usr/bin/env python3
"""Verifier workload for wro-tvm-winograd-conv2d-transform.

Two modes:

  correctness -- every public entry point of ``winograd_conv`` is compared
                 against an independent reference written straight from the
                 Apache TVM TOPI Winograd spec (winograd_util.py /
                 conv2d.py::_conv2d_winograd_nhwc_impl).  The Cook-Toom
                 matrices are additionally pinned against literal golden values
                 produced by TVM's own numpy code and against the defining
                 minimal-filtering identity.  Every array is float64 and is
                 compared at 1e-12 relative tolerance; shapes, dtypes and the
                 documented ``ValueError``s are checked too.
  timing      -- two Winograd blocks (an F(4, 3) 16 -> 24 block and an F(2, 3)
                 24 -> 16 block) measured with ``time.process_time``, min of
                 three, with a checksum guard.

Nothing here imports anything from the module under test other than the module
itself; every reference value is recomputed locally.
"""
import json
import sys
import time

import numpy as np
from numpy.polynomial import polynomial as npoly

TOKEN = "WRO_WINOGRAD_RESULT"

# ---- timing configuration -------------------------------------------------
TIM_SEED = 20260726
TIM_A = (1, 20, 20, 16, 24, 3, 1, 4)   # N, H, W, CI, CO, r, pad, tile
TIM_B = (1, 16, 16, 24, 16, 3, 1, 2)


class Fail(Exception):
    pass


def eq(tag, got, want):
    if got != want:
        raise Fail("%s: got %r, want %r" % (tag, got, want))


def arr_close(tag, got, want, rtol=1e-12, atol=1e-12):
    g = np.asarray(got)
    w = np.asarray(want)
    if g.shape != w.shape:
        raise Fail("%s: shape %s, want %s" % (tag, g.shape, w.shape))
    if g.dtype != np.float64:
        raise Fail("%s: dtype %s, want float64" % (tag, g.dtype))
    if not np.all(np.isfinite(g)):
        raise Fail("%s: contains non-finite values" % tag)
    d = np.abs(g - w)
    tol = atol + rtol * np.abs(w)
    if not np.all(d <= tol):
        i = int(np.argmax(d - tol))
        raise Fail("%s: max abs deviation %.3e at flat %d (got %.17g want %.17g)"
                   % (tag, float(d.max()), i, float(g.reshape(-1)[i]),
                      float(w.reshape(-1)[i])))


def raises(tag, fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError:
        return
    except Exception as e:  # noqa: BLE001
        raise Fail("%s: raised %s, want ValueError" % (tag, type(e).__name__))
    raise Fail("%s: did not raise" % tag)


def rnd(shape, seed, lo=-2.0, hi=2.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=shape).astype(np.float64)


# ---------------------------------------------------------------------------
# independent reference
# ---------------------------------------------------------------------------
# _interpolation_points (winograd_util.py L92), transcribed independently and
# keyed by degree.
R_POINTS = {
    3: [0, -1, 1, 1 / 2],
    4: [0, -1, 1, 1 / 2, -2],
    5: [0, -1, 1, 1 / 2, -2, -1 / 2],
    6: [0, -1, 1, 1 / 2, -1 / 2, 2, -2],
    7: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4],
    8: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4],
    9: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 3 / 4, -4 / 3],
    10: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 3 / 4, -4 / 3],
    11: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 3 / 4, -4 / 3, 1 / 4],
    12: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, -4],
    13: [0, -1, 1, 1 / 2, -1 / 2, 2, -2, -1 / 4, 4, 1 / 4, -3 / 4, 4 / 3, 3 / 4,
         -4 / 3],
}


def r_quad(padding):
    if isinstance(padding, (int, float, np.integer, np.floating)):
        vals = (padding, padding, padding, padding)
    else:
        seq = tuple(padding)
        vals = (seq[0], seq[1], seq[0], seq[1]) if len(seq) == 2 else tuple(seq)
    return tuple(int(v) for v in vals)


def r_out_size(in_size, pad_begin, pad_end, kernel_size):
    return int(in_size) + int(pad_begin) + int(pad_end) - int(kernel_size) + 1


def r_geom(batch, out_h, out_w, m):
    nh = -(-int(out_h) // int(m))
    nw = -(-int(out_w) // int(m))
    return nh, nw, int(batch) * nh * nw


def r_matrices(m, r):
    """Cook-Toom A/B/G via a route independent of the module's scalar loops.

    Uses ``numpy.polynomial.polynomial.polyfromroots`` for the Lagrange basis,
    an explicit ``np.linalg.inv`` of the diagonal ``f``, and matrix products --
    where the module expands the same closed form with Python scalar loops.
    """
    a = np.array(R_POINTS[m + r - 2], dtype=np.float64)
    alpha = m + r - 1
    q = alpha - 1
    d = np.ones(alpha, dtype=np.float64)
    for i in range(q):
        d[i] = float(np.prod([a[i] - a[k] for k in range(q) if k != i]))
    if d[0] < 0.0:
        d[0] = -d[0]
    f = np.diagflat(d)
    finv = np.linalg.inv(f)

    def vander(ncol):
        v = np.zeros((alpha, ncol), dtype=np.float64)
        for i in range(q):
            v[i] = a[i] ** np.arange(ncol, dtype=np.float64)
        v[q, ncol - 1] = 1.0
        return v

    a_mat = vander(m)
    g_mat = (vander(r).T @ finv).T

    fm = np.zeros((q, q), dtype=np.float64)
    for i in range(q):
        roots = np.array([a[k] for k in range(q) if k != i], dtype=np.float64)
        coef = npoly.polyfromroots(roots)
        fm[i] = coef[:q] / float(np.prod(a[i] - roots))
    tail = (-(a[:q] ** q)).reshape(q, 1)
    t_mat = np.concatenate([np.eye(q, dtype=np.float64), tail], axis=1)
    b_low = fm.T @ t_mat
    b_mat = np.concatenate(
        [b_low, np.eye(alpha, dtype=np.float64)[q].reshape(1, alpha)], axis=0)
    b_mat = b_mat @ f.T
    return a_mat, b_mat, g_mat


def r_tile(data, padding, m, r):
    arr = np.asarray(data, dtype=np.float64)
    n, h, w, ci = arr.shape
    pt, pl, pb, pr = r_quad(padding)
    oh = r_out_size(h, pt, pb, r)
    ow = r_out_size(w, pl, pr, r)
    alpha = m + r - 1
    nh, nw, p_total = r_geom(n, oh, ow, m)
    out = np.zeros((alpha, alpha, p_total, ci), dtype=np.float64)
    for p in range(p_total):
        bn = p // (nh * nw)
        ph = (p // nw) % nh
        pw = p % nw
        for eps in range(alpha):
            iy = ph * m + eps - pt
            for nu in range(alpha):
                ix = pw * m + nu - pl
                if 0 <= iy < h and 0 <= ix < w:
                    for c in range(ci):
                        out[eps, nu, p, c] = arr[bn, iy, ix, c]
    return out


def r_ti(tile, b_mat):
    t = np.asarray(tile, dtype=np.float64)
    alpha, _, p_total, ci = t.shape
    out = np.zeros((alpha, alpha, p_total, ci), dtype=np.float64)
    for p in range(p_total):
        for c in range(ci):
            for eps in range(alpha):
                for nu in range(alpha):
                    acc = 0.0
                    for ra in range(alpha):
                        for rb in range(alpha):
                            acc += t[ra, rb, p, c] * b_mat[ra, eps] \
                                * b_mat[rb, nu]
                    out[eps, nu, p, c] = acc
    return out


def r_tk(weight, g_mat):
    w = np.asarray(weight, dtype=np.float64)
    kh, kw, ci, co = w.shape
    alpha = g_mat.shape[0]
    out = np.zeros((alpha, alpha, co, ci), dtype=np.float64)
    for oc in range(co):
        for c in range(ci):
            for eps in range(alpha):
                for nu in range(alpha):
                    acc = 0.0
                    for rkh in range(kh):
                        for rkw in range(kw):
                            acc += w[rkh, rkw, c, oc] * g_mat[eps, rkh] \
                                * g_mat[nu, rkw]
                    out[eps, nu, oc, c] = acc
    return out


def r_bg(data_pack, kernel_pack):
    d = np.asarray(data_pack, dtype=np.float64)
    k = np.asarray(kernel_pack, dtype=np.float64)
    alpha, _, p_total, ci = d.shape
    co = k.shape[2]
    out = np.zeros((alpha, alpha, p_total, co), dtype=np.float64)
    for p in range(p_total):
        for oc in range(co):
            for eps in range(alpha):
                for nu in range(alpha):
                    acc = 0.0
                    for c in range(ci):
                        acc += d[eps, nu, p, c] * k[eps, nu, oc, c]
                    out[eps, nu, p, oc] = acc
    return out


def r_inv(bgemm, a_mat):
    g = np.asarray(bgemm, dtype=np.float64)
    alpha, _, p_total, co = g.shape
    m = a_mat.shape[1]
    out = np.zeros((m, m, p_total, co), dtype=np.float64)
    for p in range(p_total):
        for oc in range(co):
            for vh in range(m):
                for vw in range(m):
                    acc = 0.0
                    for ra in range(alpha):
                        for rb in range(alpha):
                            acc += g[ra, rb, p, oc] * a_mat[ra, vh] \
                                * a_mat[rb, vw]
                    out[vh, vw, p, oc] = acc
    return out


def r_untile(inverse, batch, out_h, out_w, m):
    y = np.asarray(inverse, dtype=np.float64)
    co = y.shape[3]
    nh, nw, _ = r_geom(batch, out_h, out_w, m)
    out = np.zeros((batch, out_h, out_w, co), dtype=np.float64)
    for bn in range(batch):
        for oc in range(co):
            for row in range(out_h):
                for col in range(out_w):
                    p = bn * nh * nw + (row // m) * nw + (col // m)
                    out[bn, row, col, oc] = y[row % m, col % m, p, oc]
    return out


def r_pipeline(data, weight, padding, tile_size):
    arr = np.asarray(data, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    r = w.shape[0]
    m = int(tile_size)
    pt, pl, pb, pr = r_quad(padding)
    n, h, ww, _ = arr.shape
    oh = r_out_size(h, pt, pb, r)
    ow = r_out_size(ww, pl, pr, r)
    a_mat, b_mat, g_mat = r_matrices(m, r)
    tiles = r_tile(arr, (pt, pl, pb, pr), m, r)
    kp = r_tk(w, g_mat)
    dp = r_ti(tiles, b_mat)
    bg = r_bg(dp, kp)
    iv = r_inv(bg, a_mat)
    out = r_untile(iv, n, oh, ow, m)
    nh, nw, p_total = r_geom(n, oh, ow, m)
    return {"out": out, "A": a_mat, "B": b_mat, "G": g_mat,
            "input_tile": tiles, "data_pack": dp, "kernel_pack": kp,
            "bgemm": bg, "inverse": iv, "alpha": m + r - 1, "tile_size": m,
            "out_h": oh, "out_w": ow, "num_tiles": p_total}


def r_direct(data, weight, padding):
    """Naive stride-1 NHWC correlation -- the quantity Winograd must reproduce."""
    arr = np.asarray(data, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    n, h, ww, ci = arr.shape
    kh, kw, _, co = w.shape
    pt, pl, pb, pr = r_quad(padding)
    oh = r_out_size(h, pt, pb, kh)
    ow = r_out_size(ww, pl, pr, kw)
    pad = np.zeros((n, h + pt + pb, ww + pl + pr, ci), dtype=np.float64)
    pad[:, pt:pt + h, pl:pl + ww, :] = arr
    out = np.zeros((n, oh, ow, co), dtype=np.float64)
    for bn in range(n):
        for row in range(oh):
            for col in range(ow):
                win = pad[bn, row:row + kh, col:col + kw, :]
                for oc in range(co):
                    out[bn, row, col, oc] = float(np.sum(win * w[:, :, :, oc]))
    return out


# ---- golden Cook-Toom matrices, produced by TVM's own numpy code path ------
GOLD = {
    (2, 3): {
        "A": [[1.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]],
        "B": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, -1.0],
              [-1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "G": [[1.0, 0.0, 0.0], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5],
              [0.0, 0.0, 1.0]],
    },
    (4, 3): {
        "A": [[1.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0],
              [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.25, 0.125],
              [1.0, -2.0, 4.0, -8.0], [0.0, 0.0, 0.0, 1.0]],
        "B": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [-1.5, 1.0, -1.0, -2.0, 0.5, 1.0],
              [-2.0, -2.5, 0.5, -1.0, -1.0, -1.5],
              [1.5, 0.5, 2.5, 2.0, -0.5, -2.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "G": [[1.0, 0.0, 0.0],
              [-0.3333333333333333, 0.3333333333333333, -0.3333333333333333],
              [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
              [-1.0666666666666667, -0.5333333333333333, -0.26666666666666666],
              [0.06666666666666667, -0.13333333333333333, 0.26666666666666666],
              [0.0, 0.0, 1.0]],
    },
    (2, 5): {
        "A": [[1.0, 0.0], [1.0, -1.0], [1.0, 1.0], [1.0, 0.5], [1.0, -2.0],
              [0.0, 1.0]],
        "B": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [-1.5, 1.0, -1.0, -2.0, 0.5, 1.0],
              [-2.0, -2.5, 0.5, -1.0, -1.0, -1.5],
              [1.5, 0.5, 2.5, 2.0, -0.5, -2.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "G": [[1.0, 0.0, 0.0, 0.0, 0.0],
              [-0.3333333333333333, 0.3333333333333333, -0.3333333333333333,
               0.3333333333333333, -0.3333333333333333],
              [0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
               0.3333333333333333, 0.3333333333333333],
              [-1.0666666666666667, -0.5333333333333333, -0.26666666666666666,
               -0.13333333333333333, -0.06666666666666667],
              [0.06666666666666667, -0.13333333333333333, 0.26666666666666666,
               -0.5333333333333333, 1.0666666666666667],
              [0.0, 0.0, 0.0, 0.0, 1.0]],
    },
    (5, 4): {
        "A": [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.25, 0.125, 0.0625],
              [1.0, -0.5, 0.25, -0.125, 0.0625], [1.0, 2.0, 4.0, 8.0, 16.0],
              [1.0, -2.0, 4.0, -8.0, 16.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
        "B": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5, -1.0],
              [-5.25, 1.0, 1.0, 4.0, 4.0, 0.25, 0.25, 0.0],
              [0.0, 4.25, -4.25, -2.5, 2.5, -2.5, 2.5, 5.25],
              [5.25, -4.25, -4.25, -5.0, -5.0, -1.25, -1.25, 0.0],
              [0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -5.25],
              [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "G": [[1.0, 0.0, 0.0, 0.0],
              [-0.2222222222222222, 0.2222222222222222, -0.2222222222222222,
               0.2222222222222222],
              [-0.2222222222222222, -0.2222222222222222, -0.2222222222222222,
               -0.2222222222222222],
              [0.7111111111111111, 0.35555555555555557, 0.17777777777777778,
               0.08888888888888889],
              [0.7111111111111111, -0.35555555555555557, 0.17777777777777778,
               -0.08888888888888889],
              [0.011111111111111112, 0.022222222222222223,
               0.044444444444444446, 0.08888888888888889],
              [0.011111111111111112, -0.022222222222222223,
               0.044444444444444446, -0.08888888888888889],
              [0.0, 0.0, 0.0, 1.0]],
    },
}


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def correctness():
    import winograd_conv as M
    checks = 0

    eq("TILE_SIZE_MIN", int(M.TILE_SIZE_MIN), 2)
    eq("TILE_SIZE_MAX", int(M.TILE_SIZE_MAX), 8)
    eq("KERNEL_SIZE_MIN", int(M.KERNEL_SIZE_MIN), 3)
    eq("KERNEL_SIZE_MAX", int(M.KERNEL_SIZE_MAX), 7)
    checks += 4

    # ---- planner: winograd_output_size, hand-traced table ----
    table = [
        ((5, 0, 0, 3), 3),
        ((5, 1, 1, 3), 5),
        ((5, 2, 2, 3), 7),
        ((1, 1, 1, 3), 1),
        ((8, 0, 0, 3), 6),
        ((8, 2, 1, 5), 7),
        ((7, 0, 0, 7), 1),
        ((20, 1, 1, 3), 20),
        ((16, 1, 1, 3), 16),
        ((10, 0, 3, 3), 11),
        ((4, 3, 0, 4), 4),
        ((9, 2, 2, 5), 9),
    ]
    for args, want in table:
        eq("winograd_output_size%s" % (args,),
           int(M.winograd_output_size(*args)), want)
        checks += 1
    # exhaustive cross-check against the count of valid window positions
    for in_size in range(1, 25):
        for pb in range(0, 4):
            for pe in range(0, 4):
                for k in range(1, 8):
                    want = r_out_size(in_size, pb, pe, k)
                    tag = "wos(%d,%d,%d,%d)" % (in_size, pb, pe, k)
                    if want < 1:
                        raises(tag, M.winograd_output_size, in_size, pb, pe, k)
                    else:
                        eq(tag, int(M.winograd_output_size(in_size, pb, pe, k)),
                           want)
                    checks += 1
    raises("wos.negpad", M.winograd_output_size, 5, -1, 0, 3)
    raises("wos.negpad2", M.winograd_output_size, 5, 0, -1, 3)
    raises("wos.zero_in", M.winograd_output_size, 0, 0, 0, 3)
    raises("wos.zero_k", M.winograd_output_size, 5, 0, 0, 0)
    checks += 4

    # ---- planner: winograd_tile_geometry ----
    for batch in (1, 2, 5):
        for oh in range(1, 20):
            for ow in range(1, 20, 3):
                for m in range(2, 9):
                    got = tuple(int(v) for v in
                                M.winograd_tile_geometry(batch, oh, ow, m))
                    eq("geom(%d,%d,%d,%d)" % (batch, oh, ow, m), got,
                       r_geom(batch, oh, ow, m))
                    checks += 1
    raises("geom.batch0", M.winograd_tile_geometry, 0, 4, 4, 2)
    raises("geom.oh0", M.winograd_tile_geometry, 1, 0, 4, 2)
    raises("geom.ow0", M.winograd_tile_geometry, 1, 4, 0, 2)
    raises("geom.tile1", M.winograd_tile_geometry, 1, 4, 4, 1)
    raises("geom.tile9", M.winograd_tile_geometry, 1, 4, 4, 9)
    checks += 5

    # ---- planner: Cook-Toom matrices ----
    for (m, r), gold in sorted(GOLD.items()):
        a_mat, b_mat, g_mat = M.winograd_transform_matrices(m, r)
        arr_close("gold.A(%d,%d)" % (m, r), a_mat,
                  np.array(gold["A"], dtype=np.float64))
        arr_close("gold.B(%d,%d)" % (m, r), b_mat,
                  np.array(gold["B"], dtype=np.float64))
        arr_close("gold.G(%d,%d)" % (m, r), g_mat,
                  np.array(gold["G"], dtype=np.float64))
        checks += 3
    for m in range(2, 9):
        for r in range(3, 8):
            alpha = m + r - 1
            a_mat, b_mat, g_mat = M.winograd_transform_matrices(m, r)
            ra, rb, rg = r_matrices(m, r)
            arr_close("mat.A(%d,%d)" % (m, r), a_mat, ra)
            arr_close("mat.B(%d,%d)" % (m, r), b_mat, rb)
            arr_close("mat.G(%d,%d)" % (m, r), g_mat, rg)
            eq("mat.shapeA(%d,%d)" % (m, r), a_mat.shape, (alpha, m))
            eq("mat.shapeB(%d,%d)" % (m, r), b_mat.shape, (alpha, alpha))
            eq("mat.shapeG(%d,%d)" % (m, r), g_mat.shape, (alpha, r))
            checks += 6
            # defining minimal-filtering identity: A^T ((G g) . (B^T d)) A is
            # the direct correlation of the alpha-tile with the r-tap filter.
            dat = rnd((alpha, alpha), 1000 + 31 * m + r)
            flt = rnd((r, r), 2000 + 31 * m + r)
            got = a_mat.T @ ((b_mat.T @ dat @ b_mat)
                             * (g_mat @ flt @ g_mat.T)) @ a_mat
            want = np.zeros((m, m), dtype=np.float64)
            for vh in range(m):
                for vw in range(m):
                    want[vh, vw] = float(
                        np.sum(dat[vh:vh + r, vw:vw + r] * flt))
            arr_close("identity(%d,%d)" % (m, r), got, want,
                      rtol=1e-9, atol=1e-9)
            checks += 1
    raises("mat.tile1", M.winograd_transform_matrices, 1, 3)
    raises("mat.tile9", M.winograd_transform_matrices, 9, 3)
    raises("mat.kernel2", M.winograd_transform_matrices, 4, 2)
    raises("mat.kernel8", M.winograd_transform_matrices, 4, 8)
    checks += 4

    # ---- stage 1: pad_and_tile ----
    tile_cfgs = [
        (1, 5, 5, 1, 0, 2, 3),
        (1, 6, 7, 3, 1, 2, 3),
        (2, 4, 4, 2, (1, 1), 4, 3),
        (1, 8, 8, 3, (2, 1, 0, 3), 4, 3),
        (1, 1, 1, 4, 2, 2, 3),
        (3, 7, 5, 2, 0, 3, 3),
        (1, 9, 9, 1, 1, 8, 3),
        (1, 7, 7, 2, 2, 2, 5),
        (1, 10, 4, 2, (0, 2, 3, 1), 6, 3),
    ]
    for idx, (n, h, w, ci, pad, m, r) in enumerate(tile_cfgs):
        x = rnd((n, h, w, ci), 300 + idx)
        got = M.pad_and_tile(x, pad, m, r)
        want = r_tile(x, pad, m, r)
        arr_close("pad_and_tile[%d]" % idx, got, want)
        if np.shares_memory(got, x):
            raise Fail("pad_and_tile[%d] aliases its input" % idx)
        checks += 2
    raises("tile.ndim", M.pad_and_tile, rnd((4, 4, 3), 9), 0, 2, 3)
    raises("tile.negpad", M.pad_and_tile, rnd((1, 4, 4, 2), 9), -1, 2, 3)
    raises("tile.badpad", M.pad_and_tile, rnd((1, 4, 4, 2), 9), (1, 1, 1), 2, 3)
    raises("tile.tile1", M.pad_and_tile, rnd((1, 4, 4, 2), 9), 0, 1, 3)
    raises("tile.kernel8", M.pad_and_tile, rnd((1, 4, 4, 2), 9), 0, 2, 8)
    raises("tile.degenerate", M.pad_and_tile, rnd((1, 2, 2, 2), 9), 0, 2, 5)
    bad = rnd((1, 4, 4, 2), 9).copy()
    bad[0, 0, 0, 0] = np.nan
    raises("tile.nonfinite", M.pad_and_tile, bad, 0, 2, 3)
    checks += 7

    # ---- stage 2: transform_input ----
    ti_cfgs = [(2, 3, 6, 2), (2, 3, 1, 1), (4, 3, 5, 2), (2, 5, 4, 2),
               (3, 3, 7, 3), (5, 4, 3, 1)]
    for idx, (m, r, p_total, ci) in enumerate(ti_cfgs):
        alpha = m + r - 1
        _, b_mat, _ = M.winograd_transform_matrices(m, r)
        _, rbm, _ = r_matrices(m, r)
        tile = rnd((alpha, alpha, p_total, ci), 400 + idx)
        arr_close("transform_input[%d]" % idx, M.transform_input(tile, b_mat),
                  r_ti(tile, rbm))
        checks += 1
    _, b4, _ = M.winograd_transform_matrices(2, 3)
    raises("ti.ndim", M.transform_input, rnd((4, 4, 3), 9), b4)
    raises("ti.axes", M.transform_input, rnd((4, 5, 3, 2), 9), b4)
    raises("ti.bshape", M.transform_input, rnd((4, 4, 3, 2), 9),
           rnd((4, 5), 9))
    raises("ti.balpha", M.transform_input, rnd((4, 4, 3, 2), 9),
           rnd((5, 5), 9))
    checks += 4

    # ---- stage 3: transform_kernel ----
    tk_cfgs = [(2, 3, 1, 1), (2, 3, 3, 4), (4, 3, 2, 3), (6, 3, 1, 2),
               (2, 5, 3, 2), (4, 5, 2, 2), (3, 7, 2, 1), (8, 3, 2, 2)]
    for idx, (m, r, ci, co) in enumerate(tk_cfgs):
        _, _, g_mat = M.winograd_transform_matrices(m, r)
        _, _, rgm = r_matrices(m, r)
        w = rnd((r, r, ci, co), 500 + idx)
        got = M.transform_kernel(w, g_mat)
        arr_close("transform_kernel[%d]" % idx, got, r_tk(w, rgm))
        eq("transform_kernel[%d].shape" % idx, got.shape,
           (m + r - 1, m + r - 1, co, ci))
        checks += 2
    _, _, g4 = M.winograd_transform_matrices(2, 3)
    raises("tk.ndim", M.transform_kernel, rnd((3, 3, 2), 9), g4)
    raises("tk.nonsquare", M.transform_kernel, rnd((3, 5, 2, 2), 9), g4)
    raises("tk.gshape", M.transform_kernel, rnd((3, 3, 2, 2), 9),
           rnd((4, 5), 9))
    checks += 3

    # ---- stage 4: batched_gemm ----
    bg_cfgs = [(4, 3, 2, 2), (4, 1, 1, 1), (6, 5, 3, 4), (6, 4, 2, 1),
               (8, 3, 1, 3), (4, 8, 4, 2), (5, 2, 2, 3)]
    for idx, (alpha, p_total, ci, co) in enumerate(bg_cfgs):
        dp = rnd((alpha, alpha, p_total, ci), 600 + idx)
        kp = rnd((alpha, alpha, co, ci), 700 + idx)
        got = M.batched_gemm(dp, kp)
        arr_close("batched_gemm[%d]" % idx, got, r_bg(dp, kp))
        eq("batched_gemm[%d].shape" % idx, got.shape,
           (alpha, alpha, p_total, co))
        checks += 2
    raises("bg.ndim", M.batched_gemm, rnd((4, 4, 3), 9), rnd((4, 4, 2, 2), 9))
    raises("bg.axes", M.batched_gemm, rnd((4, 5, 3, 2), 9),
           rnd((4, 5, 2, 2), 9))
    raises("bg.alpha", M.batched_gemm, rnd((4, 4, 3, 2), 9),
           rnd((5, 5, 2, 2), 9))
    raises("bg.ci", M.batched_gemm, rnd((4, 4, 3, 2), 9),
           rnd((4, 4, 2, 3), 9))
    checks += 4

    # ---- stage 5: inverse_transform ----
    inv_cfgs = [(2, 3, 6, 2), (2, 3, 1, 1), (4, 3, 4, 3), (2, 5, 5, 2),
                (3, 3, 3, 2), (5, 4, 2, 2), (6, 3, 2, 1)]
    for idx, (m, r, p_total, co) in enumerate(inv_cfgs):
        alpha = m + r - 1
        a_mat, _, _ = M.winograd_transform_matrices(m, r)
        ram, _, _ = r_matrices(m, r)
        bg = rnd((alpha, alpha, p_total, co), 800 + idx)
        got = M.inverse_transform(bg, a_mat)
        arr_close("inverse_transform[%d]" % idx, got, r_inv(bg, ram))
        eq("inverse_transform[%d].shape" % idx, got.shape, (m, m, p_total, co))
        checks += 2
    a4, _, _ = M.winograd_transform_matrices(2, 3)
    raises("inv.ndim", M.inverse_transform, rnd((4, 4, 3), 9), a4)
    raises("inv.axes", M.inverse_transform, rnd((4, 5, 3, 2), 9), a4)
    raises("inv.aalpha", M.inverse_transform, rnd((4, 4, 3, 2), 9),
           rnd((5, 2), 9))
    raises("inv.awide", M.inverse_transform, rnd((4, 4, 3, 2), 9),
           rnd((4, 5), 9))
    checks += 4

    # ---- stage 6: untile ----
    unt_cfgs = [(1, 4, 4, 2, 1), (1, 6, 7, 2, 3), (2, 4, 4, 4, 2),
                (1, 1, 1, 2, 2), (3, 5, 3, 3, 1), (1, 9, 9, 8, 2),
                (1, 7, 10, 4, 3)]
    for idx, (batch, oh, ow, m, co) in enumerate(unt_cfgs):
        _, _, p_total = r_geom(batch, oh, ow, m)
        iv = rnd((m, m, p_total, co), 900 + idx)
        got = M.untile(iv, batch, oh, ow, m)
        arr_close("untile[%d]" % idx, got, r_untile(iv, batch, oh, ow, m))
        eq("untile[%d].shape" % idx, got.shape, (batch, oh, ow, co))
        if np.shares_memory(got, iv):
            raise Fail("untile[%d] aliases its input" % idx)
        checks += 3
    raises("unt.ndim", M.untile, rnd((2, 2, 4), 9), 1, 4, 4, 2)
    raises("unt.tileaxes", M.untile, rnd((2, 3, 4, 2), 9), 1, 4, 4, 2)
    raises("unt.tilemismatch", M.untile, rnd((2, 2, 4, 2), 9), 1, 4, 4, 4)
    raises("unt.ptotal", M.untile, rnd((2, 2, 5, 2), 9), 1, 4, 4, 2)
    checks += 4

    # ---- pipeline: winograd_conv2d ----
    pipe_cfgs = [
        ((1, 6, 6, 2), 3, 3, 1, 2),
        ((1, 8, 8, 3), 3, 4, 1, 4),
        ((2, 5, 5, 2), 3, 2, 0, 2),
        ((1, 7, 5, 1), 3, 2, (1, 2), 3),
        ((1, 6, 8, 2), 3, 3, (2, 0, 1, 3), 4),
        ((1, 1, 1, 2), 3, 2, 2, 2),
        ((1, 9, 4, 1), 5, 2, 2, 2),
        ((1, 6, 6, 2), 3, 3, 1, 6),
    ]
    keys = ("out", "A", "B", "G", "input_tile", "data_pack", "kernel_pack",
            "bgemm", "inverse")
    for idx, (shape, r, co, pad, m) in enumerate(pipe_cfgs):
        x = rnd(shape, 1100 + idx)
        w = rnd((r, r, shape[3], co), 1200 + idx)
        got = M.winograd_conv2d(x, w, padding=pad, tile_size=m)
        want = r_pipeline(x, w, pad, m)
        eq("pipeline[%d].keys" % idx, set(got.keys()),
           set(list(keys) + ["alpha", "tile_size", "out_h", "out_w",
                             "num_tiles"]))
        checks += 1
        for k in keys:
            arr_close("pipeline[%d].%s" % (idx, k), got[k], want[k])
            checks += 1
        for k in ("alpha", "tile_size", "out_h", "out_w", "num_tiles"):
            eq("pipeline[%d].%s" % (idx, k), int(got[k]), int(want[k]))
            checks += 1
        # cross-stage compose invariant: the pipeline must be the chained stages
        a_mat, b_mat, g_mat = M.winograd_transform_matrices(m, r)
        tiles = M.pad_and_tile(x, pad, m, r)
        dp = M.transform_input(tiles, b_mat)
        kp = M.transform_kernel(w, g_mat)
        bg = M.batched_gemm(dp, kp)
        iv = M.inverse_transform(bg, a_mat)
        out = M.untile(iv, shape[0], int(got["out_h"]), int(got["out_w"]), m)
        arr_close("compose[%d]" % idx, out, got["out"])
        checks += 1
        # and it must agree with a plain direct convolution
        arr_close("direct[%d]" % idx, got["out"], r_direct(x, w, pad),
                  rtol=1e-9, atol=1e-9)
        checks += 1
    raises("pipe.ndim", M.winograd_conv2d, rnd((6, 6, 2), 9),
           rnd((3, 3, 2, 2), 9))
    raises("pipe.wndim", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((3, 3, 2), 9))
    raises("pipe.nonsquare", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((3, 5, 2, 2), 9))
    raises("pipe.cimismatch", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((3, 3, 3, 2), 9))
    raises("pipe.tile1", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((3, 3, 2, 2), 9), 0, 1)
    raises("pipe.kernel8", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((8, 8, 2, 2), 9), 0, 2)
    raises("pipe.negpad", M.winograd_conv2d, rnd((1, 6, 6, 2), 9),
           rnd((3, 3, 2, 2), 9), -1, 2)
    checks += 7

    # ---- degeneracy invariant: the smallest legal Winograd block ----
    x = rnd((1, 1, 1, 1), 1300)
    w = rnd((3, 3, 1, 1), 1400)
    got = M.winograd_conv2d(x, w, padding=2, tile_size=2)
    eq("degenerate.shape", got["out"].shape, (1, 3, 3, 1))
    eq("degenerate.alpha", int(got["alpha"]), 4)
    eq("degenerate.num_tiles", int(got["num_tiles"]), 4)
    arr_close("degenerate.out", got["out"], r_direct(x, w, 2),
              rtol=1e-9, atol=1e-9)
    checks += 4

    return checks


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def _tim_inputs():
    n, h, w, ci, co, r, _, _ = TIM_A
    xa = rnd((n, h, w, ci), TIM_SEED)
    wa = rnd((r, r, ci, co), TIM_SEED + 1, -0.5, 0.5)
    n2, h2, w2, ci2, co2, r2, _, _ = TIM_B
    xb = rnd((n2, h2, w2, ci2), TIM_SEED + 2)
    wb = rnd((r2, r2, ci2, co2), TIM_SEED + 3, -0.5, 0.5)
    return xa, wa, xb, wb


def _sweep(M, inp):
    xa, wa, xb, wb = inp
    a = M.winograd_conv2d(xa, wa, padding=TIM_A[6], tile_size=TIM_A[7])
    b = M.winograd_conv2d(xb, wb, padding=TIM_B[6], tile_size=TIM_B[7])
    return a, b


def _checksum(a, b):
    parts = []
    for res in (a, b):
        for key in ("out", "input_tile", "data_pack", "kernel_pack", "bgemm",
                    "inverse"):
            parts.append(float(np.asarray(res[key], dtype=np.float64).sum()))
    return [round(v, 6) for v in parts]


def timing():
    import winograd_conv as M
    inp = _tim_inputs()
    # WARMUP on the identical full-size input (never a shrunken proxy).
    a, b = _sweep(M, inp)
    shapes = (tuple(np.asarray(a["out"]).shape),
              tuple(np.asarray(b["out"]).shape))
    want = ((TIM_A[0], TIM_A[1], TIM_A[2], TIM_A[4]),
            (TIM_B[0], TIM_B[1], TIM_B[2], TIM_B[4]))
    if shapes != want:
        raise Fail("timing warmup produced %s, want %s" % (shapes, want))
    chk = _checksum(a, b)
    samples = []
    for _ in range(3):
        t0 = time.process_time()
        a, b = _sweep(M, inp)
        t1 = time.process_time()
        samples.append((t1 - t0) * 1000.0)
        got = (tuple(np.asarray(a["out"]).shape),
               tuple(np.asarray(b["out"]).shape))
        if got != shapes:
            raise Fail("timing run produced %s, warmup had %s" % (got, shapes))
        if _checksum(a, b) != chk:
            raise Fail("timing run checksum drifted")
    return min(samples), shapes, samples, chk


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    res = {"mode": mode}
    if mode == "correctness":
        try:
            n = correctness()
            res.update({"correctness_ok": True, "checks": n})
        except Fail as e:
            res.update({"correctness_ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            res.update({"correctness_ok": False,
                        "error": "%s: %s" % (type(e).__name__, e)})
    elif mode == "timing":
        try:
            ms, shapes, samples, chk = timing()
            res.update({"timing_ms": round(ms, 4),
                        "out_shapes": [list(s) for s in shapes],
                        "samples_ms": [round(s, 4) for s in samples],
                        "checksum": chk,
                        "cfg": [list(TIM_A), list(TIM_B)]})
        except Exception as e:  # noqa: BLE001
            res.update({"timing_ms": -1,
                        "error": "%s: %s" % (type(e).__name__, e)})
    else:
        res.update({"error": "unknown mode"})
    print(TOKEN + " " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
