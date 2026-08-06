"""Correctness suite for gradient compression with feedback — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
greps the number of CASE_PASS lines (expects 12).

The invariants are reconstruction-agnostic: a lossless full-precision compressor
(the frozen baseline) and a compact bit-packed sign compressor both satisfy every
case. What separates them is only the payload byte count (scored separately). A
compressor that drops the feedback residual update, or reconstructs a biased /
degenerate estimate, violates the multi-step invariants and fails here.
"""

import sys
import traceback

import torch

from kb_compress_harness import (
    ATOL,
    BLOCK_SIZE,
    EF_ATOL,
    EF_RTOL,
    FP32,
    RTOL,
    _assert_close,
    forbidden_vendor_guard,
    load_candidate,
    make_grad,
    rel_l2,
    wire_bytes,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _comp(buf, residual):
    return buf.to(FP32) + residual.reshape(buf.shape).to(FP32)


def _check_ef(fn_c, fn_d, buf, residual, msg=""):
    """The defining error-feedback identity: new_residual == comp - decompress."""
    payload, new_res = fn_c(buf, residual)
    q = fn_d(payload)
    if tuple(q.shape) != tuple(buf.shape):
        raise AssertionError(f"decompress shape {tuple(q.shape)} != buf {tuple(buf.shape)} {msg}")
    if tuple(new_res.shape) != tuple(buf.shape):
        raise AssertionError(f"new_residual shape {tuple(new_res.shape)} != buf {tuple(buf.shape)} {msg}")
    comp = _comp(buf, residual)
    _assert_close(new_res, comp - q, EF_RTOL, EF_ATOL, "new_residual==comp-decompress", msg)
    return payload, new_res, q


@case
def roundtrip_shape_dtype_determinism(fn):
    c, d = fn.compress, fn.decompress
    buf = make_grad((256, 1024), 100)
    res = torch.zeros_like(buf)
    payload, new_res = c(buf, res)
    q1 = d(payload)
    q2 = d(payload)
    if q1.dtype is not FP32:
        raise AssertionError(f"decompress dtype {q1.dtype} != float32")
    if not torch.equal(q1, q2):
        raise AssertionError("decompress is not deterministic")
    if wire_bytes(payload) <= 0:
        raise AssertionError("payload occupies no bytes")


@case
def ef_consistency_normal(fn):
    buf = make_grad((512, 2048), 200)
    res = torch.zeros_like(buf)
    _check_ef(fn.compress, fn.decompress, buf, res, "[normal res=0]")


@case
def ef_consistency_with_residual(fn):
    buf = make_grad((300, 700), 300)
    res = make_grad((300, 700), 301, scale=0.3)
    _check_ef(fn.compress, fn.decompress, buf, res, "[nonzero residual, non-square]")


@case
def boundary_block_nonmultiple(fn):
    # numel deliberately not a multiple of the scale-block size
    n = BLOCK_SIZE * 3 + 37
    buf = make_grad((n,), 400)
    res = make_grad((n,), 401, scale=0.5)
    _, _, q = _check_ef(fn.compress, fn.decompress, buf, res, "[partial last block]")
    if q.numel() != n:
        raise AssertionError(f"decompress numel {q.numel()} != {n}")


@case
def boundary_all_same_sign(fn):
    for dist in ("positive", "negative"):
        buf = make_grad((128, 512), 500, dist=dist)
        res = torch.zeros_like(buf)
        _, _, q = _check_ef(fn.compress, fn.decompress, buf, res, f"[all {dist}]")
        if not torch.isfinite(q).all():
            raise AssertionError(f"non-finite reconstruction for all-{dist}")


@case
def degenerate_all_zero(fn):
    buf = make_grad((128, 256), 600, dist="zeros")
    res = torch.zeros_like(buf)
    payload, new_res, q = _check_ef(fn.compress, fn.decompress, buf, res, "[all zero]")
    _assert_close(q, torch.zeros_like(buf), ATOL, ATOL, "decompress(zeros)", "[all zero]")
    _assert_close(new_res, torch.zeros_like(buf), EF_ATOL, EF_ATOL, "new_residual(zeros)", "[all zero]")


@case
def degenerate_single_and_outlier(fn):
    # 1-element buffer
    buf1 = make_grad((1,), 700)
    _check_ef(fn.compress, fn.decompress, buf1, torch.zeros_like(buf1), "[numel=1]")
    # a single dominating spike among small values
    bufo = make_grad((4096,), 701, dist="outlier")
    _, _, q = _check_ef(fn.compress, fn.decompress, bufo, torch.zeros_like(bufo), "[outlier]")
    if not torch.isfinite(q).all():
        raise AssertionError("non-finite reconstruction for outlier buffer")


@case
def error_dtype(fn):
    buf = make_grad((64, 256), 800)
    res = torch.zeros_like(buf)
    try:
        fn.compress(buf.to(torch.float16), res.to(torch.float16))
        raise AssertionError("fp16 buf did not raise TypeError")
    except TypeError:
        pass
    try:
        fn.compress(buf, res.to(torch.float16))
        raise AssertionError("fp16 residual did not raise TypeError")
    except TypeError:
        pass
    try:
        fn.compress([1.0, 2.0], res)
        raise AssertionError("non-tensor buf did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    buf = make_grad((64, 256), 900)
    try:
        fn.compress(buf, make_grad((64, 256 + 8), 901))
        raise AssertionError("residual with wrong numel did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_scale(fn):
    """Scaling the buffer by c>0 scales the reconstruction by c (scale/sign split)."""
    c = 4.0  # exact in fp32
    buf = make_grad((256, 1024), 1000)
    zero = torch.zeros_like(buf)
    q_a = fn.decompress(fn.compress(buf, zero)[0])
    q_b = fn.decompress(fn.compress((buf * c), zero)[0])
    _assert_close(q_b, q_a * c, RTOL, ATOL + 1e-3, "decompress(c*buf)==c*decompress(buf)", "[scale]")


@case
def multistep_conservation_heavytail(fn):
    """Feedback conservation over K steps against a fixed heavy-tailed target:
    sum_k decompress_k + residual_K == K * target (exact telescoping identity).
    A no-feedback compressor leaves residual==0 and transmits a constant biased
    value, so the sum equals K*q(target) != K*target."""
    K = 50
    target = make_grad((4, BLOCK_SIZE), 1100, dist="heavytail")
    res = torch.zeros_like(target)
    acc = torch.zeros_like(target)
    for _ in range(K):
        payload, res = fn.compress(target, res)
        acc = acc + fn.decompress(payload)
    lhs = acc + res
    rhs = target * K
    r = rel_l2(lhs, rhs)
    if r >= 5e-3:
        raise AssertionError(f"feedback not conserved over {K} steps: rel_l2={r:.5f}")


@case
def multistep_unbiased_converge(fn):
    """With feedback, the running mean of the reconstruction converges to the
    fixed target. A no-feedback (biased) or degenerate (near-empty payload)
    compressor stays away from the target and fails."""
    K = 128
    target = make_grad((4, BLOCK_SIZE), 1200, dist="normal")
    res = torch.zeros_like(target)
    acc = torch.zeros_like(target)
    for _ in range(K):
        payload, res = fn.compress(target, res)
        acc = acc + fn.decompress(payload)
    mean = acc / K
    r = rel_l2(mean, target)
    if r >= 0.1:
        raise AssertionError(f"running mean not unbiased after {K} steps: rel_l2={r:.4f}")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            with forbidden_vendor_guard():
                fn_case(mod)
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
