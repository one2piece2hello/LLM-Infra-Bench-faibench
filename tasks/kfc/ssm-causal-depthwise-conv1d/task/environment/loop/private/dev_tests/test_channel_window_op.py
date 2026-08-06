"""Correctness suite for the per-channel trailing-window weighted-sum contract — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).

Covers: normal (bf16 + fp32), causality (no leakage of future positions), left-pad
boundary, K=1 pointwise, per-channel independence (no cross-row mixing), bias +
activation, long-L fp16, degenerate L=1/C=1, error contract, metamorphic shift
equivariance, and an impulse-response work-evidence check.
"""

import sys
import traceback

import torch

from kb_conv_harness import (
    BF16,
    FP16,
    FP32,
    _assert_one,
    assert_close,
    forbidden_vendor_guard,
    load_candidate,
    make_bias,
    make_w,
    make_x,
    ref_channel_window_op,
    tol_for,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _run(fn, B, C, L, K, seed, dtype=BF16, use_bias=True, msg=""):
    x = make_x(B, C, L, seed, dtype=dtype)
    w = make_w(C, K, seed + 1, dtype=dtype)
    bias = make_bias(C, seed + 2, dtype=dtype) if use_bias else None
    out = fn(x, w, bias)
    ref = ref_channel_window_op(x, w, bias)
    assert_close(out, ref, dtype, msg=msg or f"[B={B},C={C},L={L},K={K},{dtype}]")


@case
def normal_bf16(fn):
    _run(fn, 2, 256, 512, 4, seed=100, dtype=BF16, use_bias=True)


@case
def normal_fp32(fn):
    # tight fp32 tolerance; no bias to exercise the None branch
    _run(fn, 2, 128, 384, 3, seed=200, dtype=FP32, use_bias=False)


@case
def causality(fn):
    """Output position t must not depend on any input position > t. Perturbing the
    input tail leaves every output before the cut unchanged."""
    B, C, L, K = 2, 64, 256, 4
    dtype = FP32
    x = make_x(B, C, L, 300, dtype=dtype)
    w = make_w(C, K, 301, dtype=dtype)
    bias = make_bias(C, 302, dtype=dtype)
    y = fn(x, w, bias)
    t0 = L // 2
    x2 = x.clone()
    x2[:, :, t0:] = make_x(B, C, L - t0, 999, dtype=dtype)  # replace the future
    y2 = fn(x2, w, bias)
    rtol, atol = tol_for(dtype)
    _assert_one(y2[:, :, :t0], y[:, :, :t0], rtol, atol, "y[:t0]", "[causality: future perturbation leaked backward]")


@case
def left_pad_boundary(fn):
    """The first K-1 outputs use zero-filled history. Verify y[..,0] analytically:
    only the last tap sees x[..,0], all earlier taps see the zero pad."""
    B, C, L, K = 1, 8, 16, 4
    dtype = FP32
    x = make_x(B, C, L, 400, dtype=dtype)
    w = make_w(C, K, 401, dtype=dtype)
    bias = make_bias(C, 402, dtype=dtype)
    y = fn(x, w, bias)
    # analytic first position: a0 = bias[c] + w[c,K-1]*x[c,0]; y0 = silu(a0)
    a0 = bias.to(FP32).view(1, C) + w.to(FP32)[:, K - 1].view(1, C) * x.to(FP32)[:, :, 0]
    y0 = a0 * torch.sigmoid(a0)
    rtol, atol = tol_for(dtype)
    _assert_one(y[:, :, 0], y0, rtol, atol, "y[..,0]", "[left-pad boundary]")
    # and the full-tensor parity
    ref = ref_channel_window_op(x, w, bias)
    assert_close(y, ref, dtype, msg="[left-pad full]")


@case
def pointwise_K1(fn):
    """K=1 is a pure pointwise op: y = silu(bias + w*x)."""
    B, C, L, K = 2, 128, 64, 1
    dtype = FP32
    x = make_x(B, C, L, 500, dtype=dtype)
    w = make_w(C, K, 501, dtype=dtype)
    bias = make_bias(C, 502, dtype=dtype)
    y = fn(x, w, bias)
    a = bias.to(FP32).view(1, C, 1) + w.to(FP32)[:, 0].view(1, C, 1) * x.to(FP32)
    expected = a * torch.sigmoid(a)
    rtol, atol = tol_for(dtype)
    _assert_one(y, expected, rtol, atol, "y", "[pointwise K=1]")


@case
def channel_independence(fn):
    """Rows never interact. Zeroing one row's weights+bias must leave every other
    row's output bit-for-bit unchanged and drive that row's output to silu(0)=0."""
    B, C, L, K = 2, 32, 128, 4
    dtype = FP32
    x = make_x(B, C, L, 600, dtype=dtype)
    w = make_w(C, K, 601, dtype=dtype)
    bias = make_bias(C, 602, dtype=dtype)
    y_full = fn(x, w, bias)
    c0 = 0
    w2 = w.clone(); w2[c0, :] = 0
    bias2 = bias.clone(); bias2[c0] = 0
    y2 = fn(x, w2, bias2)
    rtol, atol = tol_for(dtype)
    # other rows unchanged (cross-channel mixing would perturb them)
    _assert_one(y2[:, 1:, :], y_full[:, 1:, :], rtol, atol, "y[other rows]", "[channel independence]")
    # zeroed row -> silu(0) == 0 everywhere
    if float(y2[:, c0, :].abs().max()) > 1e-4:
        raise AssertionError("zeroed row did not collapse to silu(0)=0 -> cross-channel leakage")


@case
def bias_and_activation(fn):
    """Bias is added before the gating activation, and the activation is a genuine
    SiLU (not the identity)."""
    B, C, L, K = 2, 64, 96, 3
    dtype = FP32
    x = make_x(B, C, L, 700, dtype=dtype, scale=1.5)
    w = make_w(C, K, 701, dtype=dtype)
    bias = 0.7 + make_bias(C, 702, dtype=dtype)  # shift bias away from 0
    y = fn(x, w, bias)
    ref = ref_channel_window_op(x, w, bias)
    assert_close(y, ref, dtype, msg="[bias+activation parity]")
    # pre-activation (linear) result; SiLU must have changed it materially
    xf = x.to(FP32); wf = w.to(FP32)
    xp = torch.nn.functional.pad(xf, (K - 1, 0))
    a = bias.to(FP32).view(1, C, 1).clone().expand(B, C, L).clone()
    for j in range(K):
        a = a + wf[:, j].view(1, C, 1) * xp[:, :, j:j + L]
    lin_gap = float((y.to(FP32) - a).abs().mean())
    if lin_gap < 1e-3:
        raise AssertionError("output matches the pre-activation linear result -> activation dropped")


@case
def long_L_fp16(fn):
    _run(fn, 1, 256, 4096, 4, seed=800, dtype=FP16, use_bias=True)


@case
def degenerate_L1_C1(fn):
    """B=1, C=1, L=1 with K=4: only the last tap sees the single input; the rest
    read the zero pad."""
    B, C, L, K = 1, 1, 1, 4
    dtype = FP32
    x = make_x(B, C, L, 850, dtype=dtype)
    w = make_w(C, K, 851, dtype=dtype)
    bias = make_bias(C, 852, dtype=dtype)
    y = fn(x, w, bias)
    if tuple(y.shape) != (B, C, L) or y.dtype != dtype:
        raise AssertionError(f"bad output meta {tuple(y.shape)} {y.dtype}")
    ref = ref_channel_window_op(x, w, bias)
    assert_close(y, ref, dtype, msg="[degenerate L=1 C=1]")


@case
def error_contract(fn):
    B, C, L, K = 2, 32, 64, 3
    x = make_x(B, C, L, 900, dtype=BF16)
    w = make_w(C, K, 901, dtype=BF16)
    bias = make_bias(C, 902, dtype=BF16)
    # x not 3-D -> ValueError
    try:
        fn(x[0], w, bias); raise AssertionError("2-D x did not raise ValueError")
    except ValueError:
        pass
    # w first dim != C -> ValueError
    try:
        fn(x, make_w(C + 4, K, 903, dtype=BF16), bias); raise AssertionError("wrong w rows did not raise ValueError")
    except ValueError:
        pass
    # window length K < 1 (w has shape (C, 0)) -> ValueError
    try:
        fn(x, torch.zeros(C, 0, dtype=BF16, device="cuda"), bias); raise AssertionError("K=0 did not raise ValueError")
    except ValueError:
        pass
    # bias length != C -> ValueError
    try:
        fn(x, w, make_bias(C + 4, 904, dtype=BF16)); raise AssertionError("bad bias length did not raise ValueError")
    except ValueError:
        pass
    # x fp64 (non-supported dtype) -> TypeError
    try:
        fn(x.to(torch.float64), w.to(torch.float64), bias.to(torch.float64)); raise AssertionError("fp64 x did not raise TypeError")
    except TypeError:
        pass
    # w dtype mismatch -> TypeError
    try:
        fn(x, w.to(FP16), bias); raise AssertionError("mismatched w dtype did not raise TypeError")
    except TypeError:
        pass
    # bias wrong type (not a tensor / not None) -> TypeError
    try:
        fn(x, w, 1.0); raise AssertionError("scalar bias did not raise TypeError")
    except TypeError:
        pass


@case
def metamorphic_shift(fn):
    """Shift-equivariance: shifting the input right by s along the length axis
    shifts the output right by s (interior region). A non-causal or mis-indexed
    implementation breaks this."""
    B, C, L, K = 2, 64, 256, 4
    dtype = FP32
    s = 5
    x = make_x(B, C, L, 1000, dtype=dtype)
    w = make_w(C, K, 1001, dtype=dtype)
    bias = make_bias(C, 1002, dtype=dtype)
    x_sh = torch.zeros_like(x)
    x_sh[:, :, s:] = x[:, :, :L - s]
    y = fn(x, w, bias)
    y_sh = fn(x_sh, w, bias)
    rtol, atol = tol_for(dtype)
    _assert_one(y_sh[:, :, s:], y[:, :, :L - s], rtol, atol, "shifted", "[shift equivariance]")


@case
def work_evidence_impulse(fn):
    """Impulse responses pin the exact indexing. Last-tap unit weight => identity
    (y == silu(x)); first-tap unit weight => delay by K-1 (with a zero-filled head).
    A zeros stub, a pass-through without delay, or a wrong-shift all fail here."""
    B, C, L, K = 2, 32, 64, 4
    dtype = FP32
    x = make_x(B, C, L, 1100, dtype=dtype)
    rtol, atol = tol_for(dtype)
    # (a) impulse at the last tap -> no shift, bias None -> y == silu(x)
    w_last = torch.zeros(C, K, dtype=dtype, device="cuda"); w_last[:, K - 1] = 1.0
    y_a = fn(x, w_last, None)
    silu_x = x * torch.sigmoid(x)
    _assert_one(y_a, silu_x, rtol, atol, "impulse-last", "[identity]")
    # (b) impulse at the first tap -> delay by K-1, zero-filled head
    w_first = torch.zeros(C, K, dtype=dtype, device="cuda"); w_first[:, 0] = 1.0
    y_b = fn(x, w_first, None)
    shifted = torch.zeros_like(x)
    shifted[:, :, K - 1:] = x[:, :, :L - (K - 1)]
    expected = shifted * torch.sigmoid(shifted)
    _assert_one(y_b, expected, rtol, atol, "impulse-first", "[delay by K-1]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.channel_window_op
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            with forbidden_vendor_guard():
                fn_case(fn)
            torch.cuda.synchronize()
            passed += 1
            print(f"CASE_PASS {name}")
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"CASE_FAIL {name}: {reason.splitlines()[0][:300]}")
            traceback.print_exc(file=sys.stderr)
    total = len(CASES)
    print(f"CASES_PASSED={passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
