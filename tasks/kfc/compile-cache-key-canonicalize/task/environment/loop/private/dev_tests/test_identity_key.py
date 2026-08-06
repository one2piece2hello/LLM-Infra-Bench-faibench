"""Correctness suite for the request-signature identity contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library.

Every case tests a *safety* / determinism property that any acceptable identity must
satisfy: it returns a stable string, and it NEVER gives two genuinely-different
signatures the same identity. (Collapsing equivalent spellings is rewarded by the
distinct-identity count, not gated here -- the conservative baseline, which collapses
nothing, is still correct.)
"""

import sys
import traceback

from kb_identity_harness import (
    build_labeled_workload,
    find_false_merges,
    load_candidate,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def sig(op, operands, flags=None, meta=None):
    d = {"op": op, "operands": [{"shape": list(s), "dtype": dt} for s, dt in operands]}
    if flags is not None:
        d["flags"] = dict(flags)
    if meta is not None:
        d["meta"] = dict(meta)
    return d


def _key(mod, s):
    k = mod.identity_key(s)
    if not isinstance(k, str) or not k:
        raise AssertionError(f"identity_key must return a non-empty str, got {k!r}")
    return k


@case
def returns_string_for_various(mod):
    samples = [
        sig("add", [([256], "f32"), ([256], "f32")]),
        sig("matmul", [([16, 32], "f16"), ([32, 8], "f16")], flags={"precision": "high"}),
        sig("conv", [([1, 3, 8, 8], "f32"), ([4, 3, 3, 3], "f32")], flags={"layout": "channels_last"}, meta={"note": "x"}),
        {"op": "noop"},  # minimal signature (no operands / flags / meta)
    ]
    for s in samples:
        _key(mod, s)


@case
def deterministic_repeat_same_object(mod):
    s = sig("mul", [([128], "f32"), ([128], "f32")], flags={"fastmath": True})
    if _key(mod, s) != _key(mod, s):
        raise AssertionError("identity not stable across repeated calls on the same object")


@case
def deterministic_fresh_build(mod):
    a = sig("matmul", [([64, 64], "f16"), ([64, 64], "f16")], flags={"precision": "high"}, meta={"note": "a"})
    b = sig("matmul", [([64, 64], "f16"), ([64, 64], "f16")], flags={"precision": "high"}, meta={"note": "a"})
    if _key(mod, a) != _key(mod, b):
        raise AssertionError("two independently built identical signatures got different identities")


@case
def distinct_ops_distinct_keys(mod):
    a = sig("add", [([256], "f32"), ([256], "f32")])
    b = sig("mul", [([256], "f32"), ([256], "f32")])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("different ops collided")


@case
def distinct_shapes_distinct_keys(mod):
    a = sig("add", [([256], "f32"), ([256], "f32")])
    b = sig("add", [([512], "f32"), ([512], "f32")])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("different operand shapes collided")


@case
def distinct_dtypes_distinct_keys(mod):
    # normalized dtypes differ (f32 vs f16) -> genuinely different signatures.
    a = sig("add", [([256], "f32"), ([256], "f32")])
    b = sig("add", [([256], "f16"), ([256], "f16")])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("different operand dtypes collided (dtype dropped?)")


@case
def distinct_nondefault_flag_distinct_keys(mod):
    a = sig("add", [([256], "f32"), ([256], "f32")], flags={"fastmath": True})
    b = sig("add", [([256], "f32"), ([256], "f32")], flags={"fastmath": False})
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("a non-default flag value was ignored")


@case
def noncommutative_order_significant(mod):
    # matmul is order-sensitive: swapping operands is a different program.
    a = sig("matmul", [([128, 256], "f32"), ([256, 512], "f32")])
    b = sig("matmul", [([256, 512], "f32"), ([128, 256], "f32")])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("order-sensitive operands were merged across order")


@case
def distinct_operand_count_distinct_keys(mod):
    a = sig("add", [([256], "f32"), ([256], "f32")])
    b = sig("add", [([256], "f32"), ([256], "f32"), ([256], "f32")])
    if _key(mod, a) == _key(mod, b):
        raise AssertionError("signatures with a different number of operands collided")


@case
def no_false_merge_full_workload(mod):
    workload = build_labeled_workload()
    bad = find_false_merges(mod, workload)
    if bad:
        sample = sorted((sorted(v), k) for k, v in bad.items())[:3]
        raise AssertionError(f"identity shared by >1 true class (false merge): {sample}")


@case
def handles_edge_schema(mod):
    empty_ops = sig("add", [])                      # no operands
    one_op = sig("add", [([4], "f32")])             # one operand
    no_flags = {"op": "relu", "operands": [{"shape": [8], "dtype": "f32"}]}  # no flags/meta keys
    for s in (empty_ops, one_op, no_flags):
        _key(mod, s)
    if _key(mod, empty_ops) == _key(mod, one_op):
        raise AssertionError("empty-operand signature collided with a one-operand signature")


@case
def many_distinct_programs_pairwise_distinct(mod):
    programs = [
        sig("add", [([256], "f32"), ([256], "f32")]),
        sig("mul", [([256], "f32"), ([256], "f32")]),
        sig("matmul", [([16, 32], "f32"), ([32, 8], "f32")]),
        sig("sub", [([128], "f16"), ([128], "f16")]),
        sig("conv", [([1, 3, 8, 8], "f32"), ([4, 3, 3, 3], "f32")]),
        sig("max", [([64], "i32"), ([64], "i32")]),
        sig("div", [([32], "bf16"), ([32], "bf16")]),
        sig("add", [([256], "f16"), ([256], "f16")]),
    ]
    keys = [_key(mod, s) for s in programs]
    if len(set(keys)) != len(programs):
        raise AssertionError(f"expected {len(programs)} distinct identities, got {len(set(keys))}")


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
