"""Correctness suite for the frozen-linear + low-rank-correction apply — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).
"""

import sys
import traceback

import torch

from kb_lowrank_harness import (
    BF16,
    FP32,
    assert_close,
    load_candidate,
    make_base_weight,
    make_factors,
    make_tensor,
    ref_lowrank_adapter_apply,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _run(fn, lead, N, K, r, seed, dtype=BF16, scale=2.0, msg=""):
    """lead = leading (token/batch) dims tuple; the feature dim K is appended."""
    x = make_tensor((*lead, K), seed, dtype=dtype)
    W = make_base_weight(N, K, seed + 10, dtype=dtype)
    A, B = make_factors(N, K, r, seed + 20, dtype=dtype)
    out = fn(x, W, A, B, scale)
    ref = ref_lowrank_adapter_apply(x, W, A, B, scale)
    assert_close(out, ref, msg=msg or f"[shape={(*lead, K)} N={N} r={r} dtype={dtype}]")


def _close_fp32(a, b, tag, rtol=1e-4, atol=1e-4):
    ca, cb = a.to(torch.float32), b.to(torch.float32)
    if ca.shape != cb.shape:
        raise AssertionError(f"{tag}: shape {tuple(ca.shape)} vs {tuple(cb.shape)}")
    diff = (ca - cb).abs()
    tol = atol + rtol * cb.abs()
    if (diff > tol).any():
        raise AssertionError(f"{tag}: {int((diff > tol).sum())} elements out of tolerance "
                             f"(worst excess {float((diff - tol).max()):.5f})")


@case
def normal_2d_bf16(fn):
    _run(fn, (512,), N=1024, K=1024, r=16, seed=100)


@case
def normal_3d_bf16(fn):
    # leading batch/seq dims collapse over the last (feature) axis
    _run(fn, (4, 128), N=512, K=768, r=16, seed=200)


@case
def normal_fp32(fn):
    _run(fn, (384,), N=1024, K=1024, r=16, seed=300, dtype=FP32)


@case
def boundary_r1(fn):
    # smallest possible rank
    _run(fn, (256,), N=512, K=640, r=1, seed=400)


@case
def boundary_r_near_full(fn):
    # r close to min(N, K): the low-rank structure nearly degenerates to full rank
    _run(fn, (128,), N=96, K=128, r=80, seed=500)


@case
def boundary_singleton_row_and_out1(fn):
    # single token row, and a separate out=1 (N=1) probe
    _run(fn, (1,), N=768, K=1024, r=8, seed=600, msg="[single-row]")
    _run(fn, (64,), N=1, K=512, r=4, seed=610, msg="[out=1]")


@case
def degenerate_scale_zero(fn):
    # scale == 0 -> the correction vanishes -> y == x @ base_weight.T (base only)
    K, N, r = 512, 640, 8
    x = make_tensor((256, K), 700, dtype=BF16)
    W = make_base_weight(N, K, 701, dtype=BF16)
    A, B = make_factors(N, K, r, 702, dtype=BF16)
    y0 = fn(x, W, A, B, 0.0)
    ref0 = ref_lowrank_adapter_apply(x, W, A, B, 0.0)
    assert_close(y0, ref0, msg="[scale=0 base-only]")
    # sanity: a non-zero scale must actually change the output (factors matter)
    y1 = fn(x, W, A, B, 4.0)
    if torch.equal(y0.to(torch.float32), y1.to(torch.float32)):
        raise AssertionError("scale had no effect on the output")


@case
def degenerate_zero_factor_b(fn):
    # factor_b all-zero -> correction is zero -> y == base only (zero-init invariant)
    K, N, r = 512, 640, 8
    x = make_tensor((256, K), 800, dtype=BF16)
    W = make_base_weight(N, K, 801, dtype=BF16)
    A, _ = make_factors(N, K, r, 802, dtype=BF16)
    Bz = torch.zeros(N, r, dtype=BF16, device="cuda")
    y = fn(x, W, A, Bz, 3.0)
    ref = ref_lowrank_adapter_apply(x, W, A, Bz, 3.0)
    assert_close(y, ref, msg="[zero factor_b -> base]")
    # cross-check: equals the base-only path (scale=0) within tolerance
    y_base = fn(x, W, A, Bz, 0.0)
    _close_fp32(y, y_base, "zero-factor_b vs scale-0", rtol=2e-2, atol=2e-2)


@case
def error_dtype(fn):
    K, N, r = 256, 320, 8
    x = make_tensor((64, K), 900, dtype=BF16)
    W = make_base_weight(N, K, 901, dtype=BF16)
    A, B = make_factors(N, K, r, 902, dtype=BF16)
    # x fp16 (not an allowed dtype) -> TypeError
    try:
        fn(x.to(torch.float16), W.to(torch.float16), A.to(torch.float16), B.to(torch.float16), 2.0)
        raise AssertionError("fp16 x did not raise TypeError")
    except TypeError:
        pass
    # base_weight dtype mismatch -> TypeError
    try:
        fn(x, W.to(torch.float32), A, B, 2.0)
        raise AssertionError("mismatched base_weight dtype did not raise TypeError")
    except TypeError:
        pass
    # factor_a dtype mismatch -> TypeError
    try:
        fn(x, W, A.to(torch.float32), B, 2.0)
        raise AssertionError("mismatched factor_a dtype did not raise TypeError")
    except TypeError:
        pass
    # factor_b dtype mismatch -> TypeError
    try:
        fn(x, W, A, B.to(torch.float32), 2.0)
        raise AssertionError("mismatched factor_b dtype did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    K, N, r = 256, 320, 8
    x = make_tensor((64, K), 1000, dtype=BF16)
    W = make_base_weight(N, K, 1001, dtype=BF16)
    A, B = make_factors(N, K, r, 1002, dtype=BF16)
    # x not >= 2-D -> ValueError
    try:
        fn(x[0], W, A, B, 2.0)
        raise AssertionError("1-D x did not raise ValueError")
    except ValueError:
        pass
    # base_weight K mismatch -> ValueError
    try:
        fn(x, make_base_weight(N, K + 8, 1003, dtype=BF16), A, B, 2.0)
        raise AssertionError("base_weight K mismatch did not raise ValueError")
    except ValueError:
        pass
    # factor_a K mismatch -> ValueError
    try:
        Abad, _ = make_factors(N, K + 8, r, 1004, dtype=BF16)
        fn(x, W, Abad, B, 2.0)
        raise AssertionError("factor_a K mismatch did not raise ValueError")
    except ValueError:
        pass
    # factor rank mismatch: factor_a rows (r) != factor_b cols (r+3) -> ValueError
    try:
        Bbad = make_tensor((N, r + 3), 1005, dtype=BF16)
        fn(x, W, A, Bbad, 2.0)
        raise AssertionError("factor rank mismatch did not raise ValueError")
    except ValueError:
        pass
    # factor_b N mismatch -> ValueError
    try:
        Bbad2 = make_tensor((N + 8, r), 1006, dtype=BF16)
        fn(x, W, A, Bbad2, 2.0)
        raise AssertionError("factor_b N mismatch did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_scale_and_additivity(fn):
    """Two properties of the op (checked in fp32 to avoid bf16 rounding noise):
    (1) the correction is linear in ``scale``: y(2c) - y(0) == 2*(y(c) - y(0));
    (2) the whole op is linear in x (no bias): y(x1 + x2) == y(x1) + y(x2)."""
    K, N, r = 512, 512, 16
    W = make_base_weight(N, K, 1100, dtype=FP32)
    A, B = make_factors(N, K, r, 1101, dtype=FP32)
    x = make_tensor((128, K), 1102, dtype=FP32)
    # (1) scale-linearity
    y0 = fn(x, W, A, B, 0.0)
    y1 = fn(x, W, A, B, 1.0)
    y2 = fn(x, W, A, B, 2.0)
    lhs = y2.to(torch.float32) - y0.to(torch.float32)
    rhs = 2.0 * (y1.to(torch.float32) - y0.to(torch.float32))
    _close_fp32(lhs, rhs, "scale-linearity")
    # (2) input-additivity
    x1 = make_tensor((128, K), 1103, dtype=FP32)
    x2 = make_tensor((128, K), 1104, dtype=FP32)
    y_sum = fn(x1, W, A, B, 2.0).to(torch.float32) + fn(x2, W, A, B, 2.0).to(torch.float32)
    y_of_sum = fn(x1 + x2, W, A, B, 2.0).to(torch.float32)
    _close_fp32(y_sum, y_of_sum, "input-additivity")


@case
def hidden_large_NK_small_r(fn):
    # structurally different regime: large N, K with a tiny rank stresses the
    # FLOPs/peak-memory gap between the two-matmul path and a materialized [N,K] delta
    _run(fn, (256,), N=8192, K=8192, r=4, seed=1200)


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.lowrank_adapter_apply
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
