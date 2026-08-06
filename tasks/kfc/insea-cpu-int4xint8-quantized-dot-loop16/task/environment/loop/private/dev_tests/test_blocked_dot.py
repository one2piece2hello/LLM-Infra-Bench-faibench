"""Correctness suite for the block-encoded integer dot-product contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library; no GPU / tensor library required.

The candidate is scored against an INDEPENDENT reference
(kb_blockdot_harness.ref_blocked_dot), never against the live oracle.
"""

import sys
import traceback

from kb_blockdot_harness import (
    assert_scalar_close,
    load_candidate,
    pack_signed_codes,
    ref_blocked_dot,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def blk_u(signed_codes, su):
    """A packed U-block from 32 signed codes in [-8, 7] and a scale factor."""
    return (su, pack_signed_codes(signed_codes))


def blk_v(codes, sv):
    """A V-block from 32 signed integer codes and a scale factor."""
    return (sv, list(codes))


def _check(mod, u_blocks, v_blocks, expected=None, msg=""):
    ref = ref_blocked_dot(u_blocks, v_blocks)
    out = mod.blocked_dot(u_blocks, v_blocks)
    assert_scalar_close(out, ref, msg=msg or "[vs reference]")
    if expected is not None:
        assert_scalar_close(out, expected, msg=(msg or "") + " [vs explicit expected]")
    return out


@case
def normal_single_block_hand(mod):
    # U codes all +1, V codes all +2, su=sv=16/256 -> combined = 1/256.
    # isum = 32*(1*2) = 64 -> 64/256 = 0.25.
    u = [blk_u([1] * 32, 16 / 256.0)]
    v = [blk_v([2] * 32, 16 / 256.0)]
    _check(mod, u, v, expected=0.25)


@case
def normal_multi_block_matches_reference(mod):
    # several blocks with varied signed codes and companion codes.
    u, v = [], []
    for b in range(5):
        uc = [((i + b) % 16) - 8 for i in range(32)]          # signed in [-8, 7]
        vc = [((i * 7 + b * 3) % 255) - 127 for i in range(32)]  # signed in [-127, 127]
        u.append(blk_u(uc, (b * 40 + 17) / 256.0))
        v.append(blk_v(vc, (b * 30 + 11) / 256.0))
    out = _check(mod, u, v)
    if out == 0.0:
        raise AssertionError("expected a non-trivial multi-block result")


@case
def boundary_single_block(mod):
    uc = [(i % 16) - 8 for i in range(32)]
    vc = [(i - 15) for i in range(32)]
    u = [blk_u(uc, 100 / 256.0)]
    v = [blk_v(vc, 64 / 256.0)]
    _check(mod, u, v)


@case
def boundary_empty_blocks(mod):
    out = mod.blocked_dot([], [])
    assert_scalar_close(out, 0.0, msg="[empty blocks -> 0.0]")


@case
def degenerate_zero_scale_block(mod):
    # first block has scale factor 0 -> it contributes nothing; result equals the
    # dot of the remaining (single) block.
    u = [blk_u([(i % 16) - 8 for i in range(32)], 0.0),
         blk_u([(i % 16) - 8 for i in range(32)], 48 / 256.0)]
    v = [blk_v([(i % 9) - 4 for i in range(32)], 32 / 256.0),
         blk_v([(i % 13) - 6 for i in range(32)], 80 / 256.0)]
    out = _check(mod, u, v)
    only_second = ref_blocked_dot([u[1]], [v[1]])
    assert_scalar_close(out, only_second, msg="[zero-scale block must drop out]")


@case
def degenerate_all_min_code(mod):
    # all U codes = -8 (stored half-byte 0), V codes all +3, su=sv=16/256.
    # isum = 32*(-8*3) = -768 -> -768/256 = -3.0.
    u = [blk_u([-8] * 32, 16 / 256.0)]
    v = [blk_v([3] * 32, 16 / 256.0)]
    _check(mod, u, v, expected=-3.0)


@case
def sign_zero_point_applied(mod):
    # U codes all -1 (stored half-byte 7), V codes all +1, su=sv=16/256.
    # signed: isum = 32*(-1) = -32 -> -0.125.  ignoring the -8 offset (treating the
    # half-byte 7 as +7) would give 32*7 = 224 -> +0.875. The signed value is required.
    u = [blk_u([-1] * 32, 16 / 256.0)]
    v = [blk_v([1] * 32, 16 / 256.0)]
    out = _check(mod, u, v, expected=-0.125)
    if abs(out - 0.875) < 1e-6:
        raise AssertionError("used the unsigned half-byte (missing the -8 offset)")


@case
def sign_low_high_lane_pairing(mod):
    # within each byte the low code (lane 2b) is +3 and the high code (lane 2b+1) is
    # -2; companion codes are +10 on even lanes and +1 on odd lanes. Correct pairing:
    # 3*10 + (-2)*1 = 28 per byte, x16 = 448 -> 448/256 = 1.75. Swapping low/high
    # would give (-2)*10 + 3*1 = -17 per byte -> -272/256 = -1.0625.
    u = [blk_u([3, -2] * 16, 16 / 256.0)]
    v = [blk_v([10, 1] * 16, 16 / 256.0)]
    out = _check(mod, u, v, expected=1.75)
    if abs(out - (-1.0625)) < 1e-6:
        raise AssertionError("low/high codes paired with the wrong lane")


@case
def error_block_count_mismatch(mod):
    u = [blk_u([0] * 32, 0.5), blk_u([0] * 32, 0.5)]
    v = [blk_v([0] * 32, 0.5)]
    try:
        mod.blocked_dot(u, v)
    except ValueError:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"block-count mismatch raised {type(other).__name__}, expected ValueError")
    raise AssertionError("block-count mismatch did not raise ValueError")


@case
def error_packed_block_length(mod):
    u = [(0.5, [0] * 15)]   # 15 bytes, not 16
    v = [blk_v([0] * 32, 0.5)]
    try:
        mod.blocked_dot(u, v)
    except ValueError:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"short packed block raised {type(other).__name__}, expected ValueError")
    raise AssertionError("short packed block did not raise ValueError")


@case
def error_code_block_length(mod):
    u = [blk_u([0] * 32, 0.5)]
    v = [(0.5, [0] * 31)]   # 31 codes, not 32
    try:
        mod.blocked_dot(u, v)
    except ValueError:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"short code block raised {type(other).__name__}, expected ValueError")
    raise AssertionError("short code block did not raise ValueError")


@case
def metamorphic_scale_linearity(mod):
    # doubling every U scale factor (2.0 is exactly representable) doubles the result.
    uc = [(i % 16) - 8 for i in range(32)]
    vc = [(i % 21) - 10 for i in range(32)]
    u1 = [blk_u(uc, 24 / 256.0), blk_u(uc, 72 / 256.0)]
    v = [blk_v(vc, 48 / 256.0), blk_v(vc, 16 / 256.0)]
    r1 = _check(mod, u1, v)
    u2 = [blk_u(uc, 48 / 256.0), blk_u(uc, 144 / 256.0)]   # each scale x2
    r2 = mod.blocked_dot(u2, v)
    assert_scalar_close(r2, 2.0 * r1, msg="[scaling U by 2 must scale the result by 2]")


@case
def metamorphic_negate_codes(mod):
    # negating every companion code negates the result.
    uc = [(i % 16) - 8 for i in range(32)]
    vc = [(i % 31) - 15 for i in range(32)]
    u = [blk_u(uc, 60 / 256.0)]
    v = [blk_v(vc, 96 / 256.0)]
    r = _check(mod, u, v)
    v_neg = [blk_v([-c for c in vc], 96 / 256.0)]
    r_neg = mod.blocked_dot(u, v_neg)
    assert_scalar_close(r_neg, -r, msg="[negating companion codes must negate the result]")


@case
def hidden_large_matches_reference(mod):
    # many deterministic blocks; must match the independent reference exactly.
    import random
    rng = random.Random(9973)
    u, v = [], []
    for _ in range(48):
        uc = [rng.randint(-8, 7) for _ in range(32)]
        vc = [rng.randint(-127, 127) for _ in range(32)]
        u.append(blk_u(uc, rng.randint(1, 255) / 256.0))
        v.append(blk_v(vc, rng.randint(1, 255) / 256.0))
    _check(mod, u, v)


def main():
    mod = load_candidate()
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            fn_case(mod)
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
