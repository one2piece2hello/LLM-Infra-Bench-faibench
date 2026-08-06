"""Correctness suite for the fp32 SGEMM contract — 15 cases.

Contract under test:
    sgemm(A, B, C, alpha, beta) -> D = alpha * (A @ B) + beta * C
with A (M,K), B (K,N), C (M,N) all float32 CUDA; D a NEW (M,N) float32 tensor;
the input C is not modified.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 15).
"""

import sys
import traceback

import torch

from kb_sgemm_harness import (
    FP32,
    FP64,
    RTOL,
    ATOL,
    assert_close,
    load_candidate,
    make_abc,
    make_matrix,
    ref_sgemm,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _check(fn, M, N, K, seed, alpha=1.0, beta=0.0, msg=""):
    A, B, C = make_abc(M, N, K, seed)
    out = fn(A, B, C, alpha, beta)
    ref = ref_sgemm(A, B, C, alpha, beta)
    assert_close(out, ref, "D", msg or f"[M={M} N={N} K={K} alpha={alpha} beta={beta}]")


@case
def normal_square(fn):
    _check(fn, 256, 256, 256, seed=100)


@case
def normal_rect(fn):
    _check(fn, 192, 320, 256, seed=200)


@case
def alpha_beta_accumulate(fn):
    # non-trivial alpha and beta: exercises the beta*C term and scaling
    _check(fn, 224, 160, 192, seed=250, alpha=0.75, beta=1.5)


@case
def alpha_zero(fn):
    # alpha=0 -> D == beta*C, the product term must drop out cleanly
    _check(fn, 128, 96, 320, seed=280, alpha=0.0, beta=2.0)


@case
def boundary_nonmultiple(fn):
    # none of M, N, K is a multiple of a common tile width (16/32/64)
    _check(fn, 130, 70, 50, seed=300, alpha=1.0, beta=1.0)


@case
def boundary_nonmultiple_large_k(fn):
    # K not a tile multiple AND large enough to span many K-tiles with a partial
    # tail tile (a kernel that drops the final partial K-tile fails here)
    _check(fn, 96, 112, 1000, seed=350, alpha=1.0, beta=0.5)


@case
def degenerate_k1(fn):
    # K == 1 -> the product is a rank-1 outer product
    _check(fn, 64, 48, 1, seed=400, alpha=1.0, beta=1.0)


@case
def degenerate_zero(fn):
    # all-zero A, B and C -> D must be all zero for any alpha, beta
    M, N, K = 64, 64, 128
    A = torch.zeros(M, K, dtype=FP32, device="cuda")
    B = torch.zeros(K, N, dtype=FP32, device="cuda")
    C = torch.zeros(M, N, dtype=FP32, device="cuda")
    out = fn(A, B, C, 1.5, 2.0)
    if out.shape != (M, N) or out.dtype != FP32:
        raise AssertionError(f"bad D meta {tuple(out.shape)} {out.dtype}")
    if not (out == 0).all():
        raise AssertionError("all-zero inputs did not produce zero D")


@case
def identity_operand(fn):
    # A @ I == A, so with beta=0 the result is A (structural, answer-free)
    K = 128
    A, _, _ = make_abc(96, K, K, seed=500)
    ident = torch.eye(K, dtype=FP32, device="cuda")
    Czero = torch.zeros(96, K, dtype=FP32, device="cuda")
    out = fn(A, ident, Czero, 1.0, 0.0)
    assert_close(out, A, "D", "[identity-B]")


@case
def does_not_mutate_C(fn):
    # the input C must be left unchanged (the result is a fresh buffer)
    A, B, C = make_abc(80, 96, 128, seed=550)
    C_before = C.clone()
    _ = fn(A, B, C, 1.0, 3.0)
    torch.cuda.synchronize()
    if not (C == C_before).all():
        raise AssertionError("sgemm modified its input C in place")


@case
def error_dtype(fn):
    A, B, C = make_abc(64, 64, 64, seed=600)
    # fp64 A -> TypeError
    try:
        fn(A.to(FP64), B, C, 1.0, 0.0)
        raise AssertionError("fp64 A did not raise TypeError")
    except TypeError:
        pass
    # fp16 B -> TypeError
    try:
        fn(A, B.to(torch.float16), C, 1.0, 0.0)
        raise AssertionError("fp16 B did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    A, B, C = make_abc(64, 64, 64, seed=700)
    # inner-dim mismatch: A is (64,64), B_bad is (72,64) -> 64 != 72 -> ValueError
    B_bad = make_matrix((72, 64), 701, scale=0.1)
    try:
        fn(A, B_bad, C, 1.0, 0.0)
        raise AssertionError("inner-dim mismatch did not raise ValueError")
    except ValueError:
        pass
    # wrong C shape: C_bad is (64,72) but M,N=(64,64) -> ValueError
    C_bad = make_matrix((64, 72), 702, scale=1.0)
    try:
        fn(A, B, C_bad, 1.0, 1.0)
        raise AssertionError("wrong C shape did not raise ValueError")
    except ValueError:
        pass
    # 1-D operand -> ValueError
    try:
        fn(A[0], B, C, 1.0, 0.0)
        raise AssertionError("1-D A did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_scale(fn):
    """Scaling A by a constant c scales the product term by c. With beta=0,
    D(c*A) == c * D(A) (c=2 is exact in fp32)."""
    c = 2.0
    A, B, C = make_abc(128, 160, 192, seed=800)
    Czero = torch.zeros_like(C)
    y = fn(A, B, Czero, 1.0, 0.0)
    y_scaled = fn((A * c), B, Czero, 1.0, 0.0)
    ref = y * c
    assert_close(y_scaled, ref, "D", "[scale-linearity]")


@case
def metamorphic_permute_rows(fn):
    """Permuting rows of A (and the matching rows of C) permutes rows of D
    (rows are independent)."""
    A, B, C = make_abc(144, 128, 160, seed=900)
    y = fn(A, B, C, 1.0, 1.0)
    perm = torch.randperm(A.size(0), device="cuda")
    y_perm = fn(A[perm].contiguous(), B, C[perm].contiguous(), 1.0, 1.0)
    assert_close(y_perm, y[perm].contiguous(), "D", "[permute-rows]")


@case
def hidden_large_k_rect(fn):
    # large K + rectangular, non-square: stresses the float32 accumulation over a
    # long reduction on a shape the public set never shows (guards overfit)
    _check(fn, 512, 384, 4096, seed=1200, alpha=1.25, beta=0.5, msg="[large-K=4096-rect]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.sgemm
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
