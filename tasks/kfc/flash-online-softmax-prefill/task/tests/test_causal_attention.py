"""Correctness suite for the causal scaled-dot-product attention contract -- 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).
"""

import sys
import traceback

import torch

from kb_attn_harness import (
    BF16,
    FP16,
    assert_close,
    forbidden_vendor_guard,
    load_candidate,
    make_qkv,
    make_tensor,
    ref_causal_attention,
    tols,
)


def _scale(D):
    return 1.0 / (D ** 0.5)


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _parity(fn, B, H, Hk, S, D, seed, dtype=BF16, causal=True, scale=1.0, msg=""):
    q, k, v = make_qkv(B, H, Hk, S, D, seed, dtype=dtype, scale=scale)
    sc = _scale(D)
    out = fn(q, k, v, sc, causal)
    ref = ref_causal_attention(q, k, v, sc, causal)
    assert_close(out, ref, dtype, "out", msg or f"[B={B} H={H} Hk={Hk} S={S} D={D} {dtype} causal={causal}]")


@case
def normal_causal_bf16(fn):
    _parity(fn, 2, 8, 8, 256, 128, seed=100)


@case
def normal_noncausal_bf16(fn):
    _parity(fn, 2, 8, 8, 256, 128, seed=150, causal=False)


@case
def normal_gqa_bf16(fn):
    # grouped-query: 8 query heads share 2 kv heads (group size 4)
    _parity(fn, 2, 8, 2, 320, 128, seed=200)


@case
def normal_fp16(fn):
    _parity(fn, 2, 8, 4, 256, 128, seed=300, dtype=FP16)


@case
def boundary_S_nontile_D64(fn):
    # S not a multiple of a typical tile (130) and small head dim D=64
    _parity(fn, 2, 4, 4, 130, 64, seed=400)


@case
def degenerate_S1(fn):
    # single position: causal softmax over 1 key -> out == v for that head
    q, k, v = make_qkv(2, 4, 2, 1, 64, seed=500)
    sc = _scale(64)
    out = fn(q, k, v, sc, True)
    ref = ref_causal_attention(q, k, v, sc, True)
    assert_close(out, ref, BF16, "out", "[S=1]")
    # for S==1 the (grouped) output row must equal the single value row it maps to
    group = 4 // 2
    v_exp = v.repeat_interleave(group, dim=1)
    assert_close(out, v_exp.to(out.dtype), BF16, "out", "[S=1 equals value row]")


@case
def error_H_not_multiple_Hkv(fn):
    # H=6 is not a multiple of Hk=4 -> ValueError
    q = make_tensor((2, 6, 64, 64), 600, dtype=BF16)
    k = make_tensor((2, 4, 64, 64), 601, dtype=BF16)
    v = make_tensor((2, 4, 64, 64), 602, dtype=BF16)
    try:
        fn(q, k, v, _scale(64), True)
        raise AssertionError("H not a multiple of Hk did not raise ValueError")
    except ValueError:
        pass


@case
def error_dtype(fn):
    q, k, v = make_qkv(2, 4, 4, 64, 64, seed=700)
    # q fp32 -> TypeError
    try:
        fn(q.to(torch.float32), k, v, _scale(64), True)
        raise AssertionError("fp32 q did not raise TypeError")
    except TypeError:
        pass
    # k dtype mismatch -> TypeError
    try:
        fn(q, k.to(FP16), v, _scale(64), True)
        raise AssertionError("mismatched k dtype did not raise TypeError")
    except TypeError:
        pass


@case
def error_shape(fn):
    q, k, v = make_qkv(2, 4, 4, 64, 64, seed=800)
    # 3-D q -> ValueError
    try:
        fn(q[0], k[0], v[0], _scale(64), True)
        raise AssertionError("3-D q did not raise ValueError")
    except ValueError:
        pass
    # D mismatch between q and k -> ValueError
    try:
        fn(q, make_tensor((2, 4, 64, 32), 801, dtype=BF16), v, _scale(64), True)
        raise AssertionError("mismatched D did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_scale_v(fn):
    """Scaling the values by a constant c scales the output linearly (softmax
    weights are unchanged). c=4 is exact in bf16."""
    c = 4.0
    q, k, v = make_qkv(2, 4, 4, 192, 64, seed=900)
    sc = _scale(64)
    out_a = fn(q, k, v, sc, True)
    out_b = fn(q, k, (v.to(torch.float32) * c).to(BF16), sc, True)
    rtol, atol = tols(BF16)
    from kb_attn_harness import assert_close as _ac
    _ac(out_b, (out_a.to(torch.float32) * c).to(BF16), BF16, "out", "[scale-v]")


@case
def metamorphic_key_shift(fn):
    """Adding a fixed vector b to every key adds the same constant q.b to every
    logit of a given query row, so the softmax (and hence the output) is
    unchanged -- the realizable form of 'add a constant to a row's logits'."""
    q, k, v = make_qkv(2, 4, 4, 192, 64, seed=1000)
    sc = _scale(64)
    b = make_tensor((1, 1, 1, 64), 1001, dtype=BF16, scale=0.5)  # broadcast over B,Hk,S
    out_a = fn(q, k, v, sc, True)
    out_b = fn(q, (k.to(torch.float32) + b.to(torch.float32)).to(BF16), v, sc, True)
    assert_close(out_b, out_a, BF16, "out", "[key-shift invariance]")


@case
def hidden_bf16_S8192(fn):
    # structurally larger regime than the public set: long sequence, GQA
    _parity(fn, 1, 4, 2, 8192, 64, seed=1200)


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.causal_attention
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
