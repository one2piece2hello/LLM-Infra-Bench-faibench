"""Correctness suite for the chunked host-to-device transfer / compute-overlap contract — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).

Every "normal"/"hidden" case compares the candidate against ref_streamed_apply, which is
the explicit *sequential copy-then-compute* ground truth — so those cases are also the
metamorphic "result == sequential reference" check. Ordering and determinism get their
own dedicated cases.
"""

import sys
import traceback

import torch

from kb_overlap_harness import (
    BF16,
    FP16,
    assert_close,
    forbidden_overlap_guard,
    load_candidate,
    make_chunks,
    make_compute,
    ref_streamed_apply,
    total_rows,
    total_h2d_bytes,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def normal_multi_bf16(fn):
    D, F = 256, 256
    chunks = make_chunks([128, 128, 128, 128], D, seed=100, dtype=BF16)
    compute = make_compute(D, F, seed=101, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[normal-multi-bf16]")


@case
def normal_fp16(fn):
    D, F = 320, 192
    chunks = make_chunks([96, 96, 96], D, seed=200, dtype=FP16)
    compute = make_compute(D, F, seed=201, dtype=FP16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[normal-fp16]")


@case
def boundary_single_chunk(fn):
    # N=1: no overlap is possible, but the result must still be correct.
    D, F = 512, 128
    chunks = make_chunks([256], D, seed=300, dtype=BF16)
    compute = make_compute(D, F, seed=301, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[single-chunk]")


@case
def boundary_many_small(fn):
    # Many tiny chunks: a dropped tail or off-by-one in the pipeline shows up here.
    D, F = 128, 128
    chunks = make_chunks([4] * 16, D, seed=400, dtype=BF16)
    compute = make_compute(D, F, seed=401, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[many-small]")
    if out.shape[0] != 16 * 4:
        raise AssertionError(f"expected {16 * 4} output rows, got {out.shape[0]}")


@case
def boundary_varying_rows(fn):
    # Chunks with different leading (row) dimensions.
    D, F = 256, 160
    chunks = make_chunks([64, 200, 8, 128, 33], D, seed=500, dtype=BF16)
    compute = make_compute(D, F, seed=501, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[varying-rows]")


@case
def degenerate_empty(fn):
    # Empty chunk list -> empty CUDA tensor.
    D, F = 256, 256
    compute = make_compute(D, F, seed=601, dtype=BF16)
    out = fn([], compute)
    if not isinstance(out, torch.Tensor) or not out.is_cuda:
        raise AssertionError("empty input must return a CUDA tensor")
    if out.numel() != 0:
        raise AssertionError(f"empty input must return a 0-element tensor, got numel={out.numel()}")


@case
def degenerate_zero_row_chunk(fn):
    # A 0-row chunk among normal ones contributes nothing and must not corrupt order.
    D, F = 192, 192
    chunks = make_chunks([64, 0, 128], D, seed=700, dtype=BF16)
    compute = make_compute(D, F, seed=701, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[zero-row-chunk]")
    if out.shape[0] != 64 + 0 + 128:
        raise AssertionError(f"expected {64 + 128} output rows, got {out.shape[0]}")


@case
def error_type(fn):
    D, F = 128, 128
    chunks = make_chunks([8, 8], D, seed=800, dtype=BF16)
    compute = make_compute(D, F, seed=801, dtype=BF16)
    # compute not callable -> TypeError
    try:
        fn(chunks, 12345)
        raise AssertionError("non-callable compute did not raise TypeError")
    except TypeError:
        pass
    # a non-tensor chunk -> TypeError
    try:
        fn([chunks[0], "not_a_tensor"], compute)
        raise AssertionError("non-tensor chunk did not raise TypeError")
    except TypeError:
        pass
    # mixed dtype across chunks -> TypeError
    try:
        fn([chunks[0], chunks[1].to(FP16)], compute)
        raise AssertionError("mixed-dtype chunks did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    D, F = 128, 128
    chunks = make_chunks([8, 8], D, seed=900, dtype=BF16)
    compute = make_compute(D, F, seed=901, dtype=BF16)
    # a chunk already on CUDA (not host memory) -> ValueError
    try:
        fn([chunks[0], torch.randn(8, D, dtype=torch.float32).to(BF16).cuda()], compute)
        raise AssertionError("CUDA chunk did not raise ValueError")
    except ValueError:
        pass
    # inconsistent trailing shape -> ValueError
    try:
        fn([chunks[0], torch.randn(8, D + 8, dtype=torch.float32).to(BF16)], compute)
        raise AssertionError("mismatched trailing shape did not raise ValueError")
    except ValueError:
        pass
    # 0-D chunk -> ValueError
    try:
        fn([torch.tensor(1.0, dtype=BF16)], compute)
        raise AssertionError("0-D chunk did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_chunk_order(fn):
    """Results must follow INPUT chunk order, not completion order: permuting the chunk
    list permutes the output blocks correspondingly (equal-row chunks)."""
    D, F = 256, 128
    R = 64
    n = 5
    chunks = make_chunks([R] * n, D, seed=1000, dtype=BF16)
    compute = make_compute(D, F, seed=1001, dtype=BF16)
    out = fn(chunks, compute)
    perm = [3, 0, 4, 1, 2]
    out_perm = fn([chunks[i] for i in perm], compute)
    for k, src in enumerate(perm):
        a = out_perm[k * R:(k + 1) * R].to(torch.float32)
        b = out[src * R:(src + 1) * R].to(torch.float32)
        if not torch.allclose(a, b, rtol=5e-2, atol=5e-2):
            raise AssertionError(f"output block {k} did not follow input order (expected source {src})")


@case
def determinism_repeat(fn):
    # Repeated runs on identical input produce identical output.
    D, F = 256, 256
    chunks = make_chunks([100, 100, 100], D, seed=1100, dtype=BF16)
    compute = make_compute(D, F, seed=1101, dtype=BF16)
    out_a = fn(chunks, compute)
    out_b = fn(chunks, compute)
    torch.cuda.synchronize()
    if not torch.equal(out_a, out_b):
        raise AssertionError("repeated runs produced different outputs (nondeterministic)")


@case
def hidden_large_work_evidence(fn):
    # Larger multi-chunk regime (hidden). Checks parity to the sequential reference AND
    # that every chunk contributed exactly its rows (row accounting = work evidence).
    D, F = 1024, 1024
    rows_list = [512] * 8
    chunks = make_chunks(rows_list, D, seed=1200, dtype=BF16)
    compute = make_compute(D, F, seed=1201, dtype=BF16)
    out = fn(chunks, compute)
    ref = ref_streamed_apply(chunks, compute)
    assert_close(out, ref, msg="[hidden-large]")
    if out.shape[0] != total_rows(chunks):
        raise AssertionError(
            f"output rows {out.shape[0]} != sum of chunk rows {total_rows(chunks)} (a chunk was dropped/duplicated)")
    print(f"WORK_EVIDENCE h2d_bytes={total_h2d_bytes(chunks)} out_rows={out.shape[0]}")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.streamed_chunk_apply
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            with forbidden_overlap_guard():
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
