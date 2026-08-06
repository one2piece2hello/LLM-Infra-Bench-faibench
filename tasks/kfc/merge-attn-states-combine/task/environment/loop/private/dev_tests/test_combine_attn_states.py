"""Correctness suite for the partial-attention-state combine contract — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).
"""

import sys
import traceback

import torch

from kb_combine_harness import (
    BF16,
    F32,
    FP16,
    assert_pair_close,
    forbidden_vendor_guard,
    load_candidate,
    make_partials,
    ref_combine_attn_states,
    tol_for,
    _assert_one,
)

CASES = []
NEG_INF = float("-inf")


def case(fn):
    CASES.append(fn)
    return fn


def _check(fn, partial_out, partial_lse, msg=""):
    out = fn(partial_out, partial_lse)
    ref = ref_combine_attn_states(partial_out, partial_lse)
    assert_pair_close(out, ref, msg=msg or f"[shape={tuple(partial_out.shape)} dtype={partial_out.dtype}]")
    return out, ref


@case
def normal_fp32(fn):
    po, pl = make_partials(4, 4096, 128, seed=100, dtype=F32)
    _check(fn, po, pl)


@case
def normal_bf16(fn):
    po, pl = make_partials(4, 4096, 128, seed=200, dtype=BF16)
    _check(fn, po, pl)


@case
def boundary_single_chunk(fn):
    # N == 1: combined output must reproduce the single partial exactly.
    po, pl = make_partials(1, 2048, 128, seed=300, dtype=F32)
    out, _ = _check(fn, po, pl)
    cout, clse = out
    _assert_one(cout, po[0], *tol_for(F32), name="out", msg="[single-chunk==partial0]")
    _assert_one(clse, pl[0], *tol_for(F32), name="lse", msg="[single-chunk lse]", allow_nonfinite=True)


@case
def boundary_many_chunks(fn):
    # large chunk count (deep split-KV): N = 32
    po, pl = make_partials(32, 2048, 128, seed=400, dtype=F32)
    _check(fn, po, pl)


@case
def boundary_nonpow2_D(fn):
    # D not a power of two nor a common tile multiple
    po, pl = make_partials(4, 3000, 96, seed=500, dtype=F32)
    _check(fn, po, pl)


@case
def degenerate_one_dominates(fn):
    # one chunk's log-normalizer dwarfs the others -> combined output ~= that chunk.
    po, pl = make_partials(4, 2048, 128, seed=600, dtype=F32)
    pl = pl.clone()
    pl[2] += 40.0  # chunk 2 dominates
    out, _ = _check(fn, po, pl, msg="[one-chunk-dominates]")
    cout, _ = out
    # sanity: dominated combine is close to the dominating partial
    _assert_one(cout, po[2], 5e-2, 5e-2, name="out", msg="[dominant~=partial2]")


@case
def degenerate_neg_inf(fn):
    # empty chunks (-inf lse) contribute nothing; a fully empty row -> zero out, -inf lse.
    po, pl = make_partials(4, 2048, 128, seed=700, dtype=F32)
    pl = pl.clone()
    pl[0, :] = NEG_INF          # chunk 0 saw no keys anywhere
    pl[:, 5] = NEG_INF          # row 5 saw no keys in any chunk (fully empty)
    out, ref = _check(fn, po, pl, msg="[neg-inf empty chunks/row]")
    cout, clse = out
    # the fully empty row must be exactly zero and its lse exactly -inf
    if not torch.equal(cout[5].to(F32), torch.zeros(po.shape[-1], device=cout.device)):
        raise AssertionError("fully-empty row did not produce a zero output")
    if torch.isfinite(clse[5]):
        raise AssertionError("fully-empty row lse must be -inf")


@case
def error_dtype(fn):
    po, pl = make_partials(4, 256, 64, seed=800, dtype=F32)
    # non-tensor -> TypeError
    try:
        fn([1, 2, 3], pl)
        raise AssertionError("non-tensor partial_out did not raise TypeError")
    except TypeError:
        pass
    # integer dtype -> TypeError
    try:
        fn(po.to(torch.int32), pl.to(torch.int32), )
        raise AssertionError("int partials did not raise TypeError")
    except TypeError:
        pass
    # dtype mismatch between out and lse -> TypeError
    try:
        fn(po, pl.to(BF16))
        raise AssertionError("mismatched lse dtype did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    po, pl = make_partials(4, 256, 64, seed=900, dtype=F32)
    # partial_out not 3-D -> ValueError
    try:
        fn(po[0], pl)
        raise AssertionError("2-D partial_out did not raise ValueError")
    except ValueError:
        pass
    # partial_lse not 2-D -> ValueError
    try:
        fn(po, pl[0])
        raise AssertionError("1-D partial_lse did not raise ValueError")
    except ValueError:
        pass
    # (N, R) disagreement -> ValueError (wrong chunk count on lse)
    try:
        fn(po, make_partials(3, 256, 64, seed=902, dtype=F32)[1])
        raise AssertionError("N mismatch did not raise ValueError")
    except ValueError:
        pass
    # R disagreement -> ValueError
    try:
        fn(po, make_partials(4, 300, 64, seed=903, dtype=F32)[1])
        raise AssertionError("R mismatch did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_chunk_permute(fn):
    # combine is symmetric in the chunk axis -> permuting chunks leaves both outputs.
    po, pl = make_partials(6, 2048, 128, seed=1000, dtype=F32)
    y0, l0 = fn(po, pl)
    perm = torch.randperm(po.shape[0], device=po.device)
    y1, l1 = fn(po[perm], pl[perm])
    _assert_one(y1, y0, *tol_for(F32), name="out", msg="[chunk-permute]")
    _assert_one(l1, l0, *tol_for(F32), name="lse", msg="[chunk-permute]", allow_nonfinite=True)


@case
def metamorphic_lse_shift(fn):
    # adding a constant to EVERY log-normalizer leaves `out` unchanged and shifts
    # every `lse` by that constant (log-domain scale invariance of softmax weights).
    c = 2.0
    po, pl = make_partials(4, 2048, 128, seed=1100, dtype=F32)
    y0, l0 = fn(po, pl)
    y1, l1 = fn(po, pl + c)
    _assert_one(y1, y0, *tol_for(F32), name="out", msg="[lse-shift invariance]")
    _assert_one(l1, l0 + c, *tol_for(F32), name="lse", msg="[lse-shift]", allow_nonfinite=True)


@case
def hidden_bf16_H8_D256(fn):
    # structurally different regime: bf16 dtype, few chunks, large head dim.
    po, pl = make_partials(3, 4096, 256, seed=1200, dtype=BF16)
    _check(fn, po, pl)


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.combine_attn_states
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
