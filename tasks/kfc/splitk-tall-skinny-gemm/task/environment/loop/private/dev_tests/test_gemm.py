"""Correctness suite for the tall-skinny fp32 GEMM contract — 11 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 11).

The shapes emphasise the few-output / large-inner-dimension regime (small M,N
and large K) where the frozen single-block-per-tile baseline under-occupies the
device, but also cover general correctness: an awkward inner dimension that is
not a multiple of any natural partition width, a pure dot product (M=N=1), a tiny
inner dimension, an identity operand, the dtype/shape error contract, and two
metamorphic invariants. All value checks compare against a float64 reference, so
a kernel that silently drops part of the inner-dimension reduction is rejected.
"""

import sys
import traceback

import torch

from kb_gemm_harness import (
    FP32,
    FP64,
    RTOL,
    ATOL,
    assert_close,
    load_candidate,
    make_ab,
    make_matrix,
    ref_gemm,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _check(fn, M, N, K, seed, msg=""):
    A, B = make_ab(M, N, K, seed)
    out = fn(A, B)
    ref = ref_gemm(A, B)
    assert_close(out, ref, "C", msg or f"[M={M} N={N} K={K}]")


@case
def normal_tall_skinny(fn):
    # few output rows/cols, large inner dimension -> the target regime
    _check(fn, 128, 128, 32768, seed=100, msg="[tall-skinny 128x128x32768]")


@case
def normal_small_rect(fn):
    # small non-square output, large inner dimension
    _check(fn, 128, 64, 16384, seed=200, msg="[small-rect 128x64x16384]")


@case
def boundary_nonmultiple_k(fn):
    # inner dimension is odd and not a multiple of any natural partition width
    # (4/8/16/32) -> the last partition/slice is partial; small output
    _check(fn, 96, 48, 20003, seed=300, msg="[nonmultiple-K=20003]")


@case
def degenerate_dot_mn1(fn):
    # M = N = 1 -> the whole op is a single long dot product over the inner dim
    _check(fn, 1, 1, 40000, seed=400, msg="[dot M=N=1 K=40000]")


@case
def degenerate_small_k(fn):
    # inner dimension smaller than a natural partition width -> a correct kernel
    # must clamp/guard so tiny K still produces the exact product
    _check(fn, 64, 48, 3, seed=500, msg="[small-K=3]")


@case
def identity_operand(fn):
    # A @ I == A (structural: an identity right operand returns A exactly in fp32)
    K = 256
    A, _ = make_ab(96, K, K, seed=600)
    ident = torch.eye(K, dtype=FP32, device="cuda")
    out = fn(A, ident)
    assert_close(out, A, "C", "[identity-B]")


@case
def error_dtype(fn):
    A, B = make_ab(64, 64, 64, seed=700)
    # float64 A -> TypeError (contract is float32)
    try:
        fn(A.to(FP64), B)
        raise AssertionError("float64 A did not raise TypeError")
    except TypeError:
        pass
    # float64 B -> TypeError
    try:
        fn(A, B.to(FP64))
        raise AssertionError("float64 B did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    A, B = make_ab(64, 64, 64, seed=800)
    # inner-dim mismatch: A is (64,64), B_bad is (72,64) -> 64 != 72 -> ValueError
    B_bad = make_matrix((72, 64), 801, scale=0.1, dtype=FP32)
    try:
        fn(A, B_bad)
        raise AssertionError("inner-dim mismatch did not raise ValueError")
    except ValueError:
        pass
    # 1-D operand -> ValueError
    try:
        fn(A[0], B)
        raise AssertionError("1-D A did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_scale(fn):
    """Scaling A by a constant c scales C by c (c=2 is exact in fp32)."""
    c = 2.0
    A, B = make_ab(64, 96, 16384, seed=900)
    y = fn(A, B)
    y_scaled = fn((A * c), B)
    ref = y * c
    assert_close(y_scaled, ref, "C", "[scale-linearity]")


@case
def metamorphic_permute_rows(fn):
    """Permuting rows of A permutes rows of C (rows are independent)."""
    A, B = make_ab(144, 96, 8192, seed=1000)
    y = fn(A, B)
    perm = torch.randperm(A.size(0), device="cuda")
    y_perm = fn(A[perm].contiguous(), B)
    assert_close(y_perm, y[perm].contiguous(), "C", "[permute-rows]")


@case
def hidden_large_k(fn):
    # very large inner dimension, non-square small output: stresses the
    # partitioned reduction and its combine; structurally distinct from the
    # public/dev shapes.
    _check(fn, 128, 64, 65536, seed=1200, msg="[hidden large-K=65536]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.gemm
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
