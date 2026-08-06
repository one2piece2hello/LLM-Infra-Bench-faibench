"""Correctness suite for the fp16 GEMM contract — 10 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 10).
"""

import sys
import traceback

import torch

from kb_gemm_harness import (
    FP16,
    FP32,
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
def normal_square(fn):
    _check(fn, 256, 256, 256, seed=100)


@case
def normal_rect(fn):
    _check(fn, 192, 320, 256, seed=200)


@case
def boundary_nonmultiple(fn):
    # none of M, N, K is a multiple of a common tile width (16/32)
    _check(fn, 130, 70, 50, seed=300)


@case
def degenerate_k1(fn):
    # K == 1 -> C is an outer product
    _check(fn, 64, 48, 1, seed=400)


@case
def identity_operand(fn):
    # A @ I == A  (structural: an identity right operand returns A)
    K = 128
    A, _ = make_ab(96, K, K, seed=500)
    ident = torch.eye(K, dtype=FP16, device="cuda")
    out = fn(A, ident)
    assert_close(out, A, "C", "[identity-B]")


@case
def error_dtype(fn):
    A, B = make_ab(64, 64, 64, seed=600)
    # fp32 A -> TypeError
    try:
        fn(A.to(FP32), B)
        raise AssertionError("fp32 A did not raise TypeError")
    except TypeError:
        pass
    # fp32 B -> TypeError
    try:
        fn(A, B.to(FP32))
        raise AssertionError("fp32 B did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    A, B = make_ab(64, 64, 64, seed=700)
    # inner-dim mismatch: A is (64,64), B_bad is (72,64) -> 64 != 72 -> ValueError
    B_bad = make_matrix((72, 64), 701, scale=0.1, dtype=FP16)
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
    """Scaling A by a constant c scales C by c (c=2 is exact in fp16)."""
    c = 2.0
    A, B = make_ab(128, 160, 192, seed=800)
    y = fn(A, B)
    y_scaled = fn((A.to(FP32) * c).to(FP16), B)
    ref = (y.to(FP32) * c).to(FP16)
    assert_close(y_scaled, ref, "C", "[scale-linearity]")


@case
def metamorphic_permute_rows(fn):
    """Permuting rows of A permutes rows of C (rows are independent)."""
    A, B = make_ab(144, 128, 160, seed=900)
    y = fn(A, B)
    perm = torch.randperm(A.size(0), device="cuda")
    y_perm = fn(A[perm].contiguous(), B)
    assert_close(y_perm, y[perm].contiguous(), "C", "[permute-rows]")


@case
def hidden_large_k(fn):
    # large K stresses the float32 accumulation: a kernel that accumulates in
    # fp16 drifts out of tolerance here (structurally different from the public set)
    _check(fn, 512, 512, 8192, seed=1200, msg="[large-K=8192]")


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
