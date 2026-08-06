"""Correctness suite for the 2-D transpose contract — 10 cases.

The transpose is pure data movement, so every case is BITWISE-EXACT against a
reference transpose (no tolerance). Each case prints "CASE_PASS <name>" or
"CASE_FAIL <name>: <reason>". The runner greps the number of CASE_PASS lines
(expects 10).
"""

import sys
import traceback

import torch

from kb_transpose_harness import (
    FP16,
    FP32,
    assert_exact,
    load_candidate,
    make_matrix,
    ref_transpose,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _check(fn, M, N, seed, dtype=FP32, msg=""):
    x = make_matrix(M, N, seed, dtype=dtype)
    out = fn(x)
    ref = ref_transpose(x)
    assert_exact(out, ref, "y", msg or f"[M={M} N={N} dtype={dtype}]")


@case
def normal_square(fn):
    _check(fn, 512, 512, seed=100)


@case
def normal_rect(fn):
    # non-square: (M, N) -> (N, M)
    _check(fn, 384, 768, seed=200)


@case
def boundary_nonmultiple(fn):
    # neither M nor N is a multiple of a common block width (16/32) -> partial tiles
    _check(fn, 130, 70, seed=300)


@case
def boundary_thin_row(fn):
    # a single row (1, N) -> a single column (N, 1)
    _check(fn, 1, 2048, seed=400)


@case
def boundary_thin_col(fn):
    # a single column (M, 1) -> a single row (1, M)
    _check(fn, 2048, 1, seed=500)


@case
def degenerate_1x1(fn):
    # smallest possible matrix
    _check(fn, 1, 1, seed=600)


@case
def normal_fp16(fn):
    # half-precision elements, moved unchanged -> still bitwise-exact
    _check(fn, 512, 768, seed=700, dtype=FP16)


@case
def metamorphic_double_transpose(fn):
    """Transposing twice returns the original matrix exactly (T(T(x)) == x)."""
    x = make_matrix(256, 384, seed=800)
    once = fn(x)
    twice = fn(once)
    assert_exact(twice, x.contiguous(), "y", "[double-transpose==identity]")


@case
def error_reject(fn):
    x = make_matrix(64, 64, seed=900)
    # non-float dtype -> TypeError
    try:
        fn(x.to(torch.int64))
        raise AssertionError("int64 input did not raise TypeError")
    except TypeError:
        pass
    # 1-D input -> ValueError
    try:
        fn(x[0])
        raise AssertionError("1-D input did not raise ValueError")
    except ValueError:
        pass
    # 3-D input -> ValueError
    try:
        fn(x.reshape(8, 8, 64))
        raise AssertionError("3-D input did not raise ValueError")
    except ValueError:
        pass


@case
def hidden_skewed_fp16(fn):
    # structurally different regime: fp16 + a large, skewed, non-tile-multiple
    # rectangle (many partial edge tiles on the short axis) -> guards shape-isomorphic
    _check(fn, 4096, 300, seed=1200, dtype=FP16, msg="[skewed fp16 4096x300]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.transpose
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
