"""Winograd F(m, r) convolution front end for CPU / edge inference.

A direct 3x3 convolution spends ``9 * CI * CO`` multiplies per output pixel.  The
Winograd (Cook-Toom) minimal-filtering algorithm trades those multiplies for a
pair of cheap linear transforms: it cuts the activation into overlapping
``alpha x alpha`` tiles (``alpha = m + r - 1``), maps every tile and every filter
into a transform domain with the constant matrices ``B`` and ``G``, multiplies
*element-wise* in that domain -- which collapses to one batched ``CI``-deep GEMM
per transform position -- and maps the ``alpha x alpha`` result back down to an
``m x m`` output tile with ``A``.  For ``F(4, 3)`` that is ``36`` multiplies per
tile per channel pair instead of ``144``, which is why every CPU inference
runtime that cares about 3x3 convolutions ships a Winograd path.

This module is the layout-and-transform half of such a path, extracted from
Apache TVM's TOPI reference implementation:

* ``python/tvm/topi/nn/winograd_util.py``
  - ``_cook_toom_convolution`` (L37) with its inner ``_F_m`` (L40), ``_A_m``
    (L50) and ``_B_m`` (L58) builders,
  - ``_interpolation_points`` (L92) -- the published interpolation-point table,
  - ``winograd_transform_matrices`` (L164) -- the validated entry point.
* ``python/tvm/topi/nn/conv2d.py``, ``_conv2d_winograd_nhwc_impl`` (L1113):
  the tile geometry (L1184-L1187), ``kernel_pack`` (L1204), ``input_tile``
  (L1222), ``data_pack`` (L1234), ``bgemm`` (L1248), ``inverse`` (L1264) and the
  final ``conv2d_winograd`` gather (L1275).

Everything is NHWC and ``float64``:

* ``data``   is ``[N, H, W, CI]``
* ``weight`` is ``[KH, KW, CI, CO]``  (correlation, *not* flipped convolution)
* the transform-domain tensors are ``[alpha, alpha, P, CI]`` for the activation,
  ``[alpha, alpha, CO, CI]`` for the filter and ``[alpha, alpha, P, CO]`` for the
  batched GEMM, where ``P = N * nH * nW`` is the flattened tile index
* the inverse transform is ``[m, m, P, CO]`` and the output is
  ``[N, out_h, out_w, CO]``

Only ``stride = 1`` and ``dilation = 1`` are in scope -- exactly the regime in
which TVM enables the Winograd path -- and the kernel must be square.

This module is currently the SLOW-BUT-CORRECT reference path: each of the six
stages is a literal transliteration of the scalar reduction its TVM ``te.compute``
declares, one output element and one tap at a time, which is orders of magnitude
slower than the same arithmetic expressed over whole arrays.
"""

import numpy as np

# ``winograd_transform_matrices`` accepts ``1 < tile_size < 9`` and
# ``2 < kernel_size < 8`` (winograd_util.py L169-L172).
TILE_SIZE_MIN = 2
TILE_SIZE_MAX = 8
KERNEL_SIZE_MIN = 3
KERNEL_SIZE_MAX = 7

# _interpolation_points (winograd_util.py L92): row ``degree - 1`` holds the
# ``degree + 1`` interpolation points proposed for F(degree - r + 2, r).  Only
# the rows reachable from the supported (tile_size, kernel_size) range are kept,
# i.e. degree 3 .. 13.
_INTERPOLATION_POINTS = {
    3: (0.0, -1.0, 1.0, 0.5),
    4: (0.0, -1.0, 1.0, 0.5, -2.0),
    5: (0.0, -1.0, 1.0, 0.5, -2.0, -0.5),
    6: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0),
    7: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25),
    8: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 4.0),
    9: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 0.75, -4.0 / 3.0),
    10: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 4.0, 0.75, -4.0 / 3.0),
    11: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 4.0, 0.75, -4.0 / 3.0,
         0.25),
    12: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 4.0, 0.25, -0.75,
         4.0 / 3.0, -4.0),
    13: (0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -0.25, 4.0, 0.25, -0.75,
         4.0 / 3.0, 0.75, -4.0 / 3.0),
}


# --------------------------------------------------------------------------- #
# small shared helpers                                                        #
# --------------------------------------------------------------------------- #
def _as_f64(a, name):
    """Return ``a`` as a C-contiguous ``float64`` ndarray."""
    arr = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s must be finite" % name)
    return np.ascontiguousarray(arr)


def _quad(padding):
    """Normalise ``padding`` to ``(top, left, bottom, right)``.

    A scalar means all four sides, a pair ``(a, b)`` means ``(a, b, a, b)`` and a
    quad is taken verbatim.  Negative padding is rejected.
    """
    if np.isscalar(padding) or isinstance(padding, (int, float)):
        vals = (padding, padding, padding, padding)
    else:
        seq = tuple(padding)
        if len(seq) == 2:
            vals = (seq[0], seq[1], seq[0], seq[1])
        elif len(seq) == 4:
            vals = (seq[0], seq[1], seq[2], seq[3])
        else:
            raise ValueError("padding must be a scalar, a pair or a quad")
    out = []
    for v in vals:
        iv = int(v)
        if iv != v:
            raise ValueError("padding entries must be integers")
        if iv < 0:
            raise ValueError("padding entries must be non-negative")
        out.append(iv)
    return out[0], out[1], out[2], out[3]


def _check_sizes(tile_size, kernel_size):
    """Validate the F(tile_size, kernel_size) pair the way TVM does."""
    if int(tile_size) != tile_size or int(kernel_size) != kernel_size:
        raise ValueError("tile_size and kernel_size must be integers")
    tile_size = int(tile_size)
    kernel_size = int(kernel_size)
    if not TILE_SIZE_MIN <= tile_size <= TILE_SIZE_MAX:
        raise ValueError("unsupported tile size for Winograd: %d" % tile_size)
    if not KERNEL_SIZE_MIN <= kernel_size <= KERNEL_SIZE_MAX:
        raise ValueError("unsupported kernel size for Winograd: %d"
                         % kernel_size)
    return tile_size, kernel_size


def _poly_mul(p, q):
    """Multiply two polynomials given as ascending-power coefficient lists."""
    out = [0.0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            out[i + j] += pi * qj
    return out


# --------------------------------------------------------------------------- #
# planners -- cheap, graded for correctness                                   #
# --------------------------------------------------------------------------- #
def winograd_output_size(in_size, pad_begin, pad_end, kernel_size):
    """Spatial extent of a stride-1, dilation-1 convolution.

    ``(in_size + pad_begin + pad_end - kernel_size) // 1 + 1`` -- the ``H`` and
    ``W`` rebinding of ``_conv2d_winograd_nhwc_impl`` (conv2d.py L1182-L1183).
    Raises ``ValueError`` if the geometry is degenerate.
    """
    if int(in_size) != in_size or int(kernel_size) != kernel_size:
        raise ValueError("in_size and kernel_size must be integers")
    in_size = int(in_size)
    kernel_size = int(kernel_size)
    pb = int(pad_begin)
    pe = int(pad_end)
    if pb != pad_begin or pe != pad_end:
        raise ValueError("padding must be integral")
    if in_size < 1:
        raise ValueError("in_size must be positive")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if pb < 0 or pe < 0:
        raise ValueError("padding must be non-negative")
    out = in_size + pb + pe - kernel_size + 1
    if out < 1:
        raise ValueError("degenerate output extent %d" % out)
    return out


def winograd_tile_geometry(batch, out_h, out_w, tile_size):
    """Return ``(nH, nW, P)`` for an ``out_h x out_w`` output cut into tiles.

    ``nH, nW = ceil(out_h / m), ceil(out_w / m)`` and ``P = N * nH * nW``
    (conv2d.py L1184-L1185).  The flattened tile index is
    ``p = n * nH * nW + ph * nW + pw``.
    """
    if int(batch) != batch or int(out_h) != out_h or int(out_w) != out_w:
        raise ValueError("batch and output extents must be integers")
    batch = int(batch)
    out_h = int(out_h)
    out_w = int(out_w)
    if int(tile_size) != tile_size:
        raise ValueError("tile_size must be an integer")
    tile_size = int(tile_size)
    if batch < 1:
        raise ValueError("batch must be positive")
    if out_h < 1 or out_w < 1:
        raise ValueError("output extents must be positive")
    if not TILE_SIZE_MIN <= tile_size <= TILE_SIZE_MAX:
        raise ValueError("unsupported tile size for Winograd: %d" % tile_size)
    nh = (out_h + tile_size - 1) // tile_size
    nw = (out_w + tile_size - 1) // tile_size
    return nh, nw, batch * nh * nw


def winograd_transform_matrices(tile_size, kernel_size):
    """Cook-Toom transform matrices ``(A, B, G)`` for F(tile_size, kernel_size).

    ``A`` is ``[alpha, m]``, ``B`` is ``[alpha, alpha]`` and ``G`` is
    ``[alpha, r]`` with ``alpha = m + r - 1``; all three are ``float64``.  The
    reduction index is always the **first** axis of ``A`` and ``B`` and the
    **second** axis of ``G``, matching TVM's ``B[r_a, eps]``, ``G[eps, r_kh]``
    and ``A[r_a, vh]`` uses.

    Derived exactly as ``_cook_toom_convolution`` (winograd_util.py L37):
    with ``d[i] = prod_{k != i, k < alpha-1} (a[i] - a[k])`` for
    ``i < alpha - 1``, ``d[alpha-1] = 1`` and ``d[0]`` sign-normalised to be
    positive, the matrix ``f = diag(d)`` and

    * ``A[i, j] = a[i] ** j`` for ``i < alpha - 1``, last row ``e_{m-1}``
    * ``G = f^-1 . V_r``, i.e. ``G[i, j] = a[i] ** j / d[i]``
    * ``B = B_m . f`` where ``B_m`` is built from the Lagrange basis
      coefficients of ``prod_{k != i} (x - a[k])``

    Raises ``ValueError`` outside ``2 <= tile_size <= 8`` or
    ``3 <= kernel_size <= 7``.
    """
    m, r = _check_sizes(tile_size, kernel_size)
    alpha = m + r - 1
    nm1 = alpha - 1
    a = _INTERPOLATION_POINTS[m + r - 2]

    diag = [1.0] * alpha
    for i in range(nm1):
        acc = 1.0
        for k in range(nm1):
            if k != i:
                acc *= a[i] - a[k]
        diag[i] = acc
    if diag[0] < 0.0:
        diag[0] = -diag[0]

    a_mat = np.zeros((alpha, m), dtype=np.float64)
    for i in range(nm1):
        for j in range(m):
            a_mat[i, j] = a[i] ** j
    a_mat[nm1, m - 1] = 1.0

    g_mat = np.zeros((alpha, r), dtype=np.float64)
    for i in range(nm1):
        for j in range(r):
            g_mat[i, j] = a[i] ** j / diag[i]
    g_mat[nm1, r - 1] = 1.0 / diag[nm1]

    # lagrange[i][nth] = coefficient of x**nth in prod_{k != i} (x - a[k]),
    # scaled by the un-normalised product of differences.
    lagrange = []
    for i in range(nm1):
        poly = [1.0]
        for k in range(nm1):
            if k != i:
                poly = _poly_mul(poly, [-a[k], 1.0])
        denom = 1.0
        for k in range(nm1):
            if k != i:
                denom *= a[i] - a[k]
        lagrange.append([c / denom for c in poly])

    b_mat = np.zeros((alpha, alpha), dtype=np.float64)
    for nth in range(nm1):
        for j in range(nm1):
            b_mat[nth, j] = lagrange[j][nth] * diag[j]
        tail = 0.0
        for i in range(nm1):
            tail += lagrange[i][nth] * (-(a[i] ** nm1))
        b_mat[nth, nm1] = tail * diag[nm1]
    b_mat[nm1, nm1] = diag[nm1]

    return a_mat, b_mat, g_mat


# --------------------------------------------------------------------------- #
# stage 1 -- pad the activation and gather the overlapping tiles              #
# --------------------------------------------------------------------------- #
def pad_and_tile(data, padding, tile_size, kernel_size):
    """``[N, H, W, CI]`` -> ``input_tile[alpha, alpha, P, CI]``.

    Mirrors ``data_pad`` + ``input_tile`` (conv2d.py L1188-L1223):

    ``input_tile[eps, nu, p, ci] = data_pad[p // (nH*nW),
                                            ((p // nW) % nH) * m + eps,
                                            (p % nW) * m + nu, ci]``

    with ``data_pad`` row ``0`` sitting at input row ``-pad_top`` and column
    ``0`` at input column ``-pad_left``.  Every read that falls outside the real
    activation -- the explicit padding *and* the extra tail rows and columns the
    last tile of each axis needs -- contributes ``0``; that zero fill is what
    lets the transform run a full ``alpha x alpha`` tile unconditionally.

    ``padding`` is ``(pad_top, pad_left, pad_bottom, pad_right)``; a pair is read
    as ``(top, left)`` mirrored and a scalar as all four.  Returns a fresh
    ``float64`` array.  Raises ``ValueError`` on a non-4-D input, on negative
    padding, on an out-of-range ``tile_size``/``kernel_size`` and on a degenerate
    output extent.
    """
    arr = _as_f64(data, "data")
    if arr.ndim != 4:
        raise ValueError("data must be 4-D [N, H, W, CI]")
    m, r = _check_sizes(tile_size, kernel_size)
    pt, pl, pb, pr = _quad(padding)
    n, h, w, ci = arr.shape
    if n < 1 or h < 1 or w < 1 or ci < 1:
        raise ValueError("data must have a positive extent on every axis")
    oh = winograd_output_size(h, pt, pb, r)
    ow = winograd_output_size(w, pl, pr, r)
    alpha = m + r - 1
    nh, nw, p_total = winograd_tile_geometry(n, oh, ow, m)

    # degraded: one scalar store per (eps, nu, tile, channel).
    out = np.zeros((alpha, alpha, p_total, ci), dtype=np.float64)
    for eps in range(alpha):
        for nu in range(alpha):
            for p in range(p_total):
                batch = p // (nh * nw)
                ph = (p // nw) % nh
                pw = p % nw
                iy = ph * m + eps - pt
                ix = pw * m + nu - pl
                if iy < 0 or iy >= h or ix < 0 or ix >= w:
                    continue
                for c in range(ci):
                    out[eps, nu, p, c] = arr[batch, iy, ix, c]
    return out


# --------------------------------------------------------------------------- #
# stage 2 -- forward transform of the activation tiles                        #
# --------------------------------------------------------------------------- #
def transform_input(input_tile, b_mat):
    """``input_tile[alpha, alpha, P, CI]`` -> ``data_pack[alpha, alpha, P, CI]``.

    ``data_pack`` (conv2d.py L1225-L1236)::

        data_pack[eps, nu, p, ci] = sum_{r_a, r_b} input_tile[r_a, r_b, p, ci]
                                    * B[r_a, eps] * B[r_b, nu]

    i.e. ``B^T . tile . B`` on the two transform axes, left untouched on ``p``
    and ``ci``.  The reduction index is the **first** axis of ``B`` on both
    sides -- ``B`` is square, so transposing it has the right shape and the wrong
    contents.  Raises ``ValueError`` if ``input_tile`` is not 4-D, if its first
    two axes differ, or if ``b_mat`` is not the matching ``[alpha, alpha]``.
    """
    xa = _as_f64(input_tile, "input_tile")
    bm = _as_f64(b_mat, "b_mat")
    if xa.ndim != 4:
        raise ValueError("input_tile must be 4-D [alpha, alpha, P, CI]")
    alpha = xa.shape[0]
    if xa.shape[1] != alpha:
        raise ValueError("input_tile transform axes must agree")
    if bm.ndim != 2 or bm.shape != (alpha, alpha):
        raise ValueError("b_mat must be [alpha, alpha] with alpha=%d" % alpha)
    p_total, ci = xa.shape[2], xa.shape[3]

    # degraded: one scalar multiply-accumulate per (data_pack element, tap pair).
    out = np.zeros((alpha, alpha, p_total, ci), dtype=np.float64)
    for eps in range(alpha):
        for nu in range(alpha):
            for p in range(p_total):
                for c in range(ci):
                    acc = 0.0
                    for ra in range(alpha):
                        for rb in range(alpha):
                            acc += xa[ra, rb, p, c] * bm[ra, eps] * bm[rb, nu]
                    out[eps, nu, p, c] = acc
    return out


# --------------------------------------------------------------------------- #
# stage 3 -- forward transform of the filter                                  #
# --------------------------------------------------------------------------- #
def transform_kernel(weight, g_mat):
    """``[KH, KW, CI, CO]`` -> ``kernel_pack[alpha, alpha, CO, CI]``.

    ``kernel_pack`` (conv2d.py L1197-L1206)::

        kernel_pack[eps, nu, co, ci] = sum_{r_kh, r_kw} weight[r_kh, r_kw, ci, co]
                                       * G[eps, r_kh] * G[nu, r_kw]

    Note the two index swaps that make this stage easy to get wrong: the
    reduction index is the **second** axis of ``G`` (``G`` is ``[alpha, r]``, so
    it is not even square), and the output carries ``co`` **before** ``ci``,
    the transpose of the ``weight`` layout.  ``eps`` pairs with the kernel
    *row* and ``nu`` with the kernel *column*.

    Raises ``ValueError`` if ``weight`` is not 4-D, if the kernel is not square,
    or if ``g_mat`` is not ``[alpha, KH]``.
    """
    wa = _as_f64(weight, "weight")
    gm = _as_f64(g_mat, "g_mat")
    if wa.ndim != 4:
        raise ValueError("weight must be 4-D [KH, KW, CI, CO]")
    kh, kw, ci, co = wa.shape
    if kh != kw:
        raise ValueError("Winograd needs a square kernel, got %dx%d" % (kh, kw))
    if gm.ndim != 2 or gm.shape[1] != kh:
        raise ValueError("g_mat must be [alpha, %d]" % kh)
    alpha = gm.shape[0]
    if ci < 1 or co < 1:
        raise ValueError("weight must have positive channel extents")

    # degraded: one scalar multiply-accumulate per (kernel_pack element, tap pair).
    out = np.zeros((alpha, alpha, co, ci), dtype=np.float64)
    for eps in range(alpha):
        for nu in range(alpha):
            for oc in range(co):
                for c in range(ci):
                    acc = 0.0
                    for rkh in range(kh):
                        for rkw in range(kw):
                            acc += wa[rkh, rkw, c, oc] * gm[eps, rkh] \
                                * gm[nu, rkw]
                    out[eps, nu, oc, c] = acc
    return out


# --------------------------------------------------------------------------- #
# stage 4 -- the batched transform-domain GEMM                                #
# --------------------------------------------------------------------------- #
def batched_gemm(data_pack, kernel_pack):
    """``[alpha, alpha, P, CI]`` x ``[alpha, alpha, CO, CI]`` -> ``[alpha, alpha, P, CO]``.

    ``bgemm`` (conv2d.py L1238-L1249)::

        bgemm[eps, nu, p, co] = sum_ci data_pack[eps, nu, p, ci]
                                * kernel_pack[eps, nu, co, ci]

    One independent ``[P, CI] x [CI, CO]`` product per transform position: the
    element-wise multiply of the Winograd domain has already been folded into
    the position index, so nothing couples ``(eps, nu)`` to anything else.  This
    stage and ``transform_input`` dominate the benchmark cost.

    Raises ``ValueError`` unless both operands are 4-D, their transform axes
    agree and are square, and their trailing ``CI`` extents match.
    """
    da = _as_f64(data_pack, "data_pack")
    ka = _as_f64(kernel_pack, "kernel_pack")
    if da.ndim != 4 or ka.ndim != 4:
        raise ValueError("data_pack and kernel_pack must both be 4-D")
    alpha = da.shape[0]
    if da.shape[1] != alpha:
        raise ValueError("data_pack transform axes must agree")
    if ka.shape[0] != alpha or ka.shape[1] != alpha:
        raise ValueError("kernel_pack transform axes must match data_pack")
    if da.shape[3] != ka.shape[3]:
        raise ValueError("data_pack and kernel_pack disagree on CI: %d vs %d"
                         % (da.shape[3], ka.shape[3]))
    p_total, ci = da.shape[2], da.shape[3]
    co = ka.shape[2]

    # degraded: one scalar multiply-accumulate per (bgemm element, input channel).
    out = np.zeros((alpha, alpha, p_total, co), dtype=np.float64)
    for eps in range(alpha):
        for nu in range(alpha):
            for p in range(p_total):
                for oc in range(co):
                    acc = 0.0
                    for c in range(ci):
                        acc += da[eps, nu, p, c] * ka[eps, nu, oc, c]
                    out[eps, nu, p, oc] = acc
    return out


# --------------------------------------------------------------------------- #
# stage 5 -- inverse transform back to output tiles                            #
# --------------------------------------------------------------------------- #
def inverse_transform(bgemm_out, a_mat):
    """``bgemm[alpha, alpha, P, CO]`` -> ``inverse[m, m, P, CO]``.

    ``inverse`` (conv2d.py L1253-L1266)::

        inverse[vh, vw, p, co] = sum_{r_a, r_b} bgemm[r_a, r_b, p, co]
                                 * A[r_a, vh] * A[r_b, vw]

    i.e. ``A^T . tile . A``, shrinking each ``alpha x alpha`` transform tile to
    the ``m x m`` output tile it encodes.  ``vh`` is the tile **row** and pairs
    with the *first* ``A`` factor -- swapping ``vh`` and ``vw`` has the right
    shape and transposes every output tile.

    Raises ``ValueError`` if ``bgemm_out`` is not 4-D with square transform axes
    or if ``a_mat`` is not ``[alpha, m]`` with ``m <= alpha``.
    """
    ga = _as_f64(bgemm_out, "bgemm")
    am = _as_f64(a_mat, "a_mat")
    if ga.ndim != 4:
        raise ValueError("bgemm must be 4-D [alpha, alpha, P, CO]")
    alpha = ga.shape[0]
    if ga.shape[1] != alpha:
        raise ValueError("bgemm transform axes must agree")
    if am.ndim != 2 or am.shape[0] != alpha:
        raise ValueError("a_mat must be [alpha, m] with alpha=%d" % alpha)
    m = am.shape[1]
    if m < 1 or m > alpha:
        raise ValueError("a_mat tile extent %d out of range" % m)
    p_total, co = ga.shape[2], ga.shape[3]

    # degraded: one scalar multiply-accumulate per (inverse element, tap pair).
    out = np.zeros((m, m, p_total, co), dtype=np.float64)
    for vh in range(m):
        for vw in range(m):
            for p in range(p_total):
                for oc in range(co):
                    acc = 0.0
                    for ra in range(alpha):
                        for rb in range(alpha):
                            acc += ga[ra, rb, p, oc] * am[ra, vh] * am[rb, vw]
                    out[vh, vw, p, oc] = acc
    return out


# --------------------------------------------------------------------------- #
# stage 6 -- scatter the output tiles back to NHWC                            #
# --------------------------------------------------------------------------- #
def untile(inverse, batch, out_h, out_w, tile_size):
    """``inverse[m, m, P, CO]`` -> ``output[N, out_h, out_w, CO]``.

    The final gather (conv2d.py L1269-L1276)::

        output[n, h, w, co] = inverse[h % m, w % m,
                                      n * nH * nW + (h // m) * nW + (w // m), co]

    The tile grid is walked **row-major with stride ``nW``** and the position
    inside a tile is the *remainder*, not the quotient; a column-major tile walk
    has the right shape and the wrong contents.  When ``out_h`` or ``out_w`` is
    not a multiple of ``m`` the trailing rows and columns of the last tile are
    simply dropped.

    Returns a fresh ``float64`` array.  Raises ``ValueError`` if ``inverse`` is
    not 4-D with two equal leading axes of length ``tile_size``, or if its ``P``
    is not ``batch * nH * nW``.
    """
    ya = _as_f64(inverse, "inverse")
    if ya.ndim != 4:
        raise ValueError("inverse must be 4-D [m, m, P, CO]")
    if int(tile_size) != tile_size:
        raise ValueError("tile_size must be an integer")
    m = int(tile_size)
    if ya.shape[0] != m or ya.shape[1] != m:
        raise ValueError("inverse tile axes must both be %d" % m)
    nh, nw, p_total = winograd_tile_geometry(batch, out_h, out_w, m)
    if ya.shape[2] != p_total:
        raise ValueError("inverse has P=%d, geometry needs %d"
                         % (ya.shape[2], p_total))
    batch = int(batch)
    out_h = int(out_h)
    out_w = int(out_w)
    co = ya.shape[3]

    # degraded: one scalar read per (batch, row, column, output channel).
    out = np.empty((batch, out_h, out_w, co), dtype=np.float64)
    for bn in range(batch):
        for y in range(out_h):
            for x in range(out_w):
                p = bn * nh * nw + (y // m) * nw + (x // m)
                for oc in range(co):
                    out[bn, y, x, oc] = ya[y % m, x % m, p, oc]
    return out


# --------------------------------------------------------------------------- #
# pipeline                                                                    #
# --------------------------------------------------------------------------- #
def winograd_conv2d(data, weight, padding=0, tile_size=4):
    """Full NHWC Winograd F(tile_size, r) convolution.

    Chains ``winograd_transform_matrices``, ``pad_and_tile``,
    ``transform_kernel``, ``transform_input``, ``batched_gemm``,
    ``inverse_transform`` and ``untile`` exactly as
    ``_conv2d_winograd_nhwc_impl`` does, with ``stride = 1`` and
    ``dilation = 1``.

    Returns a dict with ``"out"``, the three transform matrices ``"A"``, ``"B"``
    and ``"G"``, the five intermediates ``"input_tile"``, ``"data_pack"``,
    ``"kernel_pack"``, ``"bgemm"`` and ``"inverse"``, and the plan values
    ``"alpha"``, ``"tile_size"``, ``"out_h"``, ``"out_w"``, ``"num_tiles"``.
    """
    arr = _as_f64(data, "data")
    wa = _as_f64(weight, "weight")
    if arr.ndim != 4:
        raise ValueError("data must be 4-D [N, H, W, CI]")
    if wa.ndim != 4:
        raise ValueError("weight must be 4-D [KH, KW, CI, CO]")
    kh, kw = wa.shape[0], wa.shape[1]
    if kh != kw:
        raise ValueError("Winograd needs a square kernel, got %dx%d" % (kh, kw))
    if wa.shape[2] != arr.shape[3]:
        raise ValueError("weight CI=%d does not match data CI=%d"
                         % (wa.shape[2], arr.shape[3]))
    m, r = _check_sizes(tile_size, kh)
    pt, pl, pb, pr = _quad(padding)
    n, h, w, _ = arr.shape
    oh = winograd_output_size(h, pt, pb, r)
    ow = winograd_output_size(w, pl, pr, r)
    nh, nw, p_total = winograd_tile_geometry(n, oh, ow, m)
    a_mat, b_mat, g_mat = winograd_transform_matrices(m, r)

    tiles = pad_and_tile(arr, (pt, pl, pb, pr), m, r)
    kernel_pack = transform_kernel(wa, g_mat)
    data_pack = transform_input(tiles, b_mat)
    bg = batched_gemm(data_pack, kernel_pack)
    inv = inverse_transform(bg, a_mat)
    out = untile(inv, n, oh, ow, m)
    return {
        "out": out,
        "A": a_mat,
        "B": b_mat,
        "G": g_mat,
        "input_tile": tiles,
        "data_pack": data_pack,
        "kernel_pack": kernel_pack,
        "bgemm": bg,
        "inverse": inv,
        "alpha": m + r - 1,
        "tile_size": m,
        "out_h": oh,
        "out_w": ow,
        "num_tiles": p_total,
    }
