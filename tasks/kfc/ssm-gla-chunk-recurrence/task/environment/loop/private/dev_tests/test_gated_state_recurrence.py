"""Correctness suite for the gated running-state sequence mixer — 12 cases.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>".
The runner greps the number of CASE_PASS lines (expects 12).

Parity cases compare the candidate against the high-precision float64 sequential
reference (both the output and, where requested, the final state). Two
metamorphic cases (prefix causality, state threading across a split) check the
candidate against itself and need no reference.
"""

import sys
import traceback

import torch

from kb_gsr_harness import (
    BF16,
    FP16,
    RTOL,
    ATOL,
    _assert_one,
    assert_output_close,
    forbidden_vendor_guard,
    load_candidate,
    make_qkv,
    make_state,
    ref_gated_state_recurrence,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _parity(fn, B, H, L, Dk, Dv, seed, dtype=BF16, gate_mode="rand",
            initial_state=None, msg="", expect_state=True):
    q, k, v, g = make_qkv(B, H, L, Dk, Dv, seed, dtype=dtype, gate_mode=gate_mode)
    out = fn(q, k, v, g, initial_state=initial_state, output_final_state=expect_state)
    ref = ref_gated_state_recurrence(q, k, v, g, initial_state=initial_state, output_final_state=expect_state)
    assert_output_close(out, ref, msg=msg or f"[B={B} H={H} L={L} Dk={Dk} Dv={Dv} {dtype} gate={gate_mode}]",
                        expect_state=expect_state)


@case
def normal_bf16(fn):
    _parity(fn, 2, 4, 64, 64, 64, seed=100)


@case
def normal_fp16_rectangular(fn):
    # fp16 + Dk != Dv (rectangular state) — different regime from the bf16 set
    _parity(fn, 2, 4, 64, 48, 64, seed=200, dtype=FP16)


@case
def normal_long_L(fn):
    # longer sequence stresses the recurrence length / chunk segmenting
    _parity(fn, 2, 2, 256, 64, 64, seed=300)


@case
def boundary_L1(fn):
    # single position: S = decay*0 + outer(k, v); o = (scale q)^T outer(k, v)
    _parity(fn, 2, 4, 1, 64, 64, seed=400)


@case
def boundary_initial_state(fn):
    # non-zero starting state, checked on output + final state (state hand-off)
    B, H, Dk, Dv = 2, 4, 64, 96
    st = make_state(B, H, Dk, Dv, seed=501)
    _parity(fn, B, H, 96, Dk, Dv, seed=500, initial_state=st, msg="[nonzero-initial-state]")


@case
def degenerate_no_decay(fn):
    # g == 0 -> exp(g) == 1 -> no decay: the state accumulates every key/value
    # outer product (a plain cumulative running sum). A stub that gates the new
    # term instead of the state agrees here but diverges under real decay.
    _parity(fn, 2, 4, 64, 64, 64, seed=600, gate_mode="zero", msg="[no-decay g==0]")


@case
def degenerate_strong_decay(fn):
    # strongly negative gate -> the state nearly resets each step, so o_t depends
    # almost only on position t's own key/value. Discriminates "gate the state"
    # from "gate the new pair".
    _parity(fn, 2, 4, 96, 64, 64, seed=700, gate_mode="strong", msg="[strong-decay]")


@case
def degenerate_v_zero(fn):
    # v == 0 with zero initial state -> the state stays zero -> output is exactly 0
    B, H, L, Dk, Dv = 2, 4, 48, 64, 64
    q, k, _, g = make_qkv(B, H, L, Dk, Dv, seed=800)
    v = torch.zeros(B, H, L, Dv, dtype=BF16, device="cuda")
    o, st = fn(q, k, v, g, output_final_state=True)
    if o.shape != (B, H, L, Dv) or o.dtype != BF16:
        raise AssertionError(f"bad o meta {tuple(o.shape)} {o.dtype}")
    if not (o.to(torch.float32) == 0).all():
        raise AssertionError("zero value did not produce zero output")
    if not (st.to(torch.float32) == 0).all():
        raise AssertionError("zero value did not leave the state at zero")


@case
def error_contract(fn):
    B, H, L, Dk, Dv = 2, 4, 16, 64, 64
    q, k, v, g = make_qkv(B, H, L, Dk, Dv, seed=900)
    # non-floating / fp32 q -> TypeError
    try:
        fn(q.to(torch.float32), k, v, g)
        raise AssertionError("fp32 q did not raise TypeError")
    except TypeError:
        pass
    # k dtype mismatch -> TypeError
    try:
        fn(q, k.to(FP16), v, g)
        raise AssertionError("mismatched k dtype did not raise TypeError")
    except TypeError:
        pass
    # k last-dim mismatch (Dk) -> ValueError
    try:
        fn(q, make_qkv(B, H, L, Dk + 8, Dv, seed=901)[1], v, g)
        raise AssertionError("mismatched k dim did not raise ValueError")
    except ValueError:
        pass
    # g wrong shape -> ValueError
    try:
        fn(q, k, v, make_qkv(B, H, L, Dk + 8, Dv, seed=902)[3])
        raise AssertionError("mismatched g shape did not raise ValueError")
    except ValueError:
        pass
    # initial_state wrong shape -> ValueError
    try:
        fn(q, k, v, g, initial_state=make_state(B, H, Dk + 4, Dv, seed=903))
        raise AssertionError("mismatched initial_state shape did not raise ValueError")
    except ValueError:
        pass


@case
def metamorphic_prefix_causality(fn):
    """Causality: outputs for the first L1 positions are unchanged when the
    sequence is extended past L1."""
    B, H, L, Dk, Dv = 2, 4, 128, 64, 64
    L1 = 48
    q, k, v, g = make_qkv(B, H, L, Dk, Dv, seed=1000)
    o_full = fn(q, k, v, g)
    o_pre = fn(q[:, :, :L1].contiguous(), k[:, :, :L1].contiguous(),
               v[:, :, :L1].contiguous(), g[:, :, :L1].contiguous())
    _assert_one(o_pre, o_full[:, :, :L1], RTOL, ATOL, "o", "[prefix-causality]")


@case
def metamorphic_state_threading(fn):
    """Chunk-boundary invariance: running [0:m] then feeding its final state into
    [m:L] reproduces the full-sequence output (the block partition must not change
    the value)."""
    B, H, L, Dk, Dv = 2, 4, 128, 64, 64
    m = 64
    q, k, v, g = make_qkv(B, H, L, Dk, Dv, seed=1100)
    o1, s1 = fn(q[:, :, :m].contiguous(), k[:, :, :m].contiguous(),
                v[:, :, :m].contiguous(), g[:, :, :m].contiguous(), output_final_state=True)
    o2 = fn(q[:, :, m:].contiguous(), k[:, :, m:].contiguous(),
            v[:, :, m:].contiguous(), g[:, :, m:].contiguous(), initial_state=s1)
    o_full = fn(q, k, v, g)
    stitched = torch.cat([o1, o2], dim=2)
    _assert_one(stitched, o_full, RTOL, ATOL, "o", "[state-threading]")


@case
def hidden_mixed_decay(fn):
    # structurally different regime: wider-spread gates (some features keep, some
    # forget fast) + longer L + fp16, outside the public mild-decay random regime.
    # Output-only parity (the scored quantity); the final state under the extreme
    # mixed decay is left to tolerance tuning on the target hardware.
    _parity(fn, 2, 2, 384, 64, 64, seed=1200, dtype=FP16, gate_mode="mixed",
            expect_state=False, msg="[mixed-decay hidden]")


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    mod = load_candidate()
    fn = mod.gated_state_recurrence
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
