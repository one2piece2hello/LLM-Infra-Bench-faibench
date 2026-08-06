"""Correctness suite for the first-order state-space scan contract — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).
"""

import sys
import traceback

import torch

from kb_scan_harness import (
    BF16,
    F32,
    FP16,
    analytic_proxy,
    assert_close,
    assert_dtype_shape,
    forbidden_scan_guard,
    load_candidate,
    make_inputs,
    ref_state_space_scan,
    tol_for,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _run(fn, Bsz, L, D, N, seed, dtype=F32, a_low=0.05, a_high=0.95, a_const=None, msg=""):
    A, B, C, x = make_inputs(Bsz, L, D, N, seed, dtype=dtype, a_low=a_low, a_high=a_high, a_const=a_const)
    with forbidden_scan_guard():
        y = fn(A, B, C, x)
    ref = ref_state_space_scan(A, B, C, x)
    rtol, atol = tol_for(dtype)
    assert_dtype_shape(y, dtype, (Bsz, L, D), msg=msg)
    assert_close(y, ref, rtol, atol, msg=msg or f"[B={Bsz} L={L} D={D} N={N} dtype={dtype}]")
    return y


@case
def normal_f32(fn):
    # spec normal: B=2, L=512, D=256, N=16, A in (0,1) stable, X random
    _run(fn, 2, 512, 256, 16, seed=100)


@case
def normal_shape_variant(fn):
    # different (B, L, D, N) exercises the batching / broadcast paths
    _run(fn, 3, 256, 128, 8, seed=200)


@case
def boundary_L1(fn):
    # single timestep: h[0] == bx[0], y[0] == C[0] . bx[0]
    _run(fn, 2, 1, 64, 8, seed=300)


@case
def boundary_nonpow2_L(fn):
    # L not a power of two must match the (internally padded) reference
    _run(fn, 2, 513, 96, 12, seed=400)


@case
def degenerate_A_zero(fn):
    # A == 0 -> memoryless: h[t] == bx[t]
    _run(fn, 2, 128, 64, 8, seed=500, a_const=0.0)


@case
def degenerate_A_one(fn):
    # A == 1 -> the state is the running prefix-sum of the input drive
    A, B, C, x = make_inputs(2, 256, 64, 8, seed=600, a_const=1.0)
    with forbidden_scan_guard():
        y = fn(A, B, C, x)
    ref = ref_state_space_scan(A, B, C, x)
    assert_close(y, ref, *tol_for(F32), msg="[A==1 prefix-sum]")
    # cross-check the prefix-sum identity directly: h == cumsum(bx) along time
    bx = (x.to(torch.float64).unsqueeze(-1) * B.to(torch.float64).unsqueeze(2))
    h_cumsum = torch.cumsum(bx, dim=1)                                   # (B, L, D, N)
    y_id = (h_cumsum * C.to(torch.float64).unsqueeze(2)).sum(dim=-1)     # (B, L, D)
    assert_close(y, y_id, *tol_for(F32), msg="[A==1 == cumsum identity]")


@case
def error_dtype(fn):
    A, B, C, x = make_inputs(2, 32, 16, 4, seed=700)
    # non-floating x -> TypeError
    try:
        fn(A, B, C, x.to(torch.int32))
        raise AssertionError("integer x did not raise TypeError")
    except TypeError:
        pass
    # dtype mismatch among inputs -> TypeError
    try:
        fn(A.to(torch.float16), B, C, x)
        raise AssertionError("mismatched A dtype did not raise TypeError")
    except TypeError:
        pass
    # non-tensor -> TypeError
    try:
        fn(A, B, C, 3.0)
        raise AssertionError("non-tensor x did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    A, B, C, x = make_inputs(2, 32, 16, 4, seed=800)
    # x wrong rank -> ValueError
    try:
        fn(A, B, C, x[:, :, 0])
        raise AssertionError("2-D x did not raise ValueError")
    except ValueError:
        pass
    # A leading dims inconsistent with x -> ValueError
    try:
        fn(A[:, :16], B, C, x)
        raise AssertionError("A/x L-mismatch did not raise ValueError")
    except ValueError:
        pass
    # B state width inconsistent with A's N -> ValueError
    try:
        fn(A, B[:, :, :2], C, x)
        raise AssertionError("B N-mismatch did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_causality(fn):
    """State-order-preserving: y[:, :t0] must be independent of inputs at steps >= t0
    (the recurrence is causal). Perturb a future timestep and assert the earlier
    outputs are unchanged."""
    Bsz, L, D, N, t0 = 2, 192, 96, 12, 100
    A, B, C, x = make_inputs(Bsz, L, D, N, seed=900)
    with forbidden_scan_guard():
        y_a = fn(A, B, C, x)
    x2 = x.clone()
    x2[:, t0:] += 7.0                                   # perturb the future only
    B2 = B.clone()
    B2[:, t0:] *= -3.0
    with forbidden_scan_guard():
        y_b = fn(A, B2, C, x2)
    # earlier outputs identical (deterministic causal algorithm)
    assert_close(y_b[:, :t0], y_a[:, :t0].to(torch.float64),
                 1e-4, 1e-5, msg="[causality: y[:t0] changed under a future perturbation]")


@case
def metamorphic_prefix_consistency(fn):
    """Running the scan on the truncated prefix [:, :t0] must reproduce y[:, :t0] of
    the full run (within tolerance)."""
    Bsz, L, D, N, t0 = 2, 320, 64, 8, 137
    A, B, C, x = make_inputs(Bsz, L, D, N, seed=1000)
    with forbidden_scan_guard():
        y_full = fn(A, B, C, x)
        y_pref = fn(A[:, :t0], B[:, :t0], C[:, :t0], x[:, :t0])
    assert_dtype_shape(y_pref, F32, (Bsz, t0, D), msg="[prefix]")
    assert_close(y_pref, y_full[:, :t0].to(torch.float64),
                 *tol_for(F32), msg="[prefix-consistency]")


@case
def error_amplify_no_clamp(fn):
    """A > 1 amplifies the state; the candidate must accumulate the true (large)
    growth and match the reference, not silently clamp/saturate."""
    _run(fn, 2, 64, 48, 8, seed=1100, a_const=1.08, msg="[amplifying A>1]")


@case
def hidden_bf16_large_L(fn):
    # reviewer-stressed regime: long L + bf16 (state accumulated in fp32); guards S7.
    proxy = analytic_proxy(2, 4096, 128, 64)
    assert proxy["seq_depth"] == 4096
    _run(fn, 2, 4096, 128, 64, seed=1200, dtype=BF16, msg="[bf16 long-L]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.state_space_scan
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
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
