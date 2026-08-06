"""Correctness suite for the bounded integer index-expression simplifier.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library; no GPU / torch.

The candidate is judged against an INDEPENDENT reference (kb_symbolic_harness:
an expression evaluator + value-equivalence checker), never against the live
oracle. The central gate is semantic: the simplified tree must evaluate to the
same value as the input for EVERY assignment of the variables over their declared
ranges (exhaustive small-range enumeration).
"""

import sys
import traceback

from kb_symbolic_harness import (
    C,
    V,
    eval_expr,
    free_vars,
    load_candidate,
    node_count,
    recombine_pattern,
    values_equivalent,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _simplify_equiv(mod, expr, bounds, msg=""):
    """Simplify and assert (a) value-equivalence to the input over the bounded
    domain and (b) the tree did not grow. Returns the simplified tree."""
    out = mod.simplify_expr(expr, bounds)
    if not values_equivalent(out, expr, bounds):
        raise AssertionError(f"simplified tree not value-equivalent to input {msg}")
    if node_count(out) > node_count(expr):
        raise AssertionError(
            f"simplification grew the tree {node_count(expr)} -> {node_count(out)} {msg}")
    return out


@case
def normal_recombine_pow2(mod):
    # (i%4) + (i//4)*4 == i for all integer i
    expr = recombine_pattern("i", 4)
    _simplify_equiv(mod, expr, {"i": (0, 63)}, msg="[recombine pow2]")


@case
def normal_trivial_identities(mod):
    # ((x*1) + 0) == x
    expr = ("add", ("mul", V("x"), C(1)), C(0))
    _simplify_equiv(mod, expr, {"x": (0, 10)}, msg="[x*1+0]")


@case
def normal_const_fold(mod):
    # (3*4) + 5 == 17, no variables
    expr = ("add", ("mul", C(3), C(4)), C(5))
    out = _simplify_equiv(mod, expr, {}, msg="[const fold]")
    if eval_expr(out, {}) != 17:
        raise AssertionError(f"expected 17, evaluates to {eval_expr(out, {})}")


@case
def boundary_single_var(mod):
    # a bare variable is already minimal; must come back a single node, unchanged value
    expr = V("x")
    out = _simplify_equiv(mod, expr, {"x": (0, 7)}, msg="[single var]")
    if node_count(out) != 1:
        raise AssertionError(f"single var must stay 1 node, got {node_count(out)}")


@case
def boundary_range_width_1(mod):
    # variable pinned to one value (lo == hi)
    expr = ("add", V("x"), C(0))
    _simplify_equiv(mod, expr, {"x": (5, 5)}, msg="[width-1 range]")


@case
def degenerate_already_canonical(mod):
    # a + b has no legal reduction; it must return value-equal and not grow
    expr = ("add", V("a"), V("b"))
    _simplify_equiv(mod, expr, {"a": (0, 9), "b": (0, 9)}, msg="[already canonical]")


@case
def error_div_by_zero_const(mod):
    # floor division by the constant 0 -> defined rejection
    expr = ("floordiv", V("x"), C(0))
    try:
        mod.simplify_expr(expr, {"x": (0, 5)})
    except ValueError:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"div-by-zero raised {type(other).__name__}, expected ValueError")
    raise AssertionError("div by constant 0 did not raise ValueError")


@case
def error_unbounded_var(mod):
    # a variable with no declared bound -> defined rejection
    expr = ("add", V("x"), V("z"))
    try:
        mod.simplify_expr(expr, {"x": (0, 5)})
    except ValueError:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(f"unbounded var raised {type(other).__name__}, expected ValueError")
    raise AssertionError("unbounded variable did not raise ValueError")


@case
def metamorphic_idempotent(mod):
    # simplifying an already-simplified tree must not change its value or grow it
    expr = ("mul", ("add", recombine_pattern("i", 8), C(0)), C(1))
    bounds = {"i": (0, 63)}
    once = mod.simplify_expr(expr, bounds)
    twice = mod.simplify_expr(once, bounds)
    if not values_equivalent(twice, once, bounds):
        raise AssertionError("re-simplifying changed the value (not idempotent)")
    if node_count(twice) > node_count(once):
        raise AssertionError(
            f"re-simplifying grew the tree {node_count(once)} -> {node_count(twice)}")


@case
def metamorphic_value_equivalence_exhaustive(mod):
    # THE semantic gate: a mixed tree checked over the full bounded domain
    expr = ("add", recombine_pattern("a", 6), ("mod", V("b"), C(3)))
    bounds = {"a": (0, 35), "b": (0, 20)}   # 36 * 21 = 756 assignments -> exhaustive
    _simplify_equiv(mod, expr, bounds, msg="[exhaustive value-equivalence]")


@case
def hidden_nonpow2_recombine(mod):
    # non-power-of-two modulus: a bit-masking shortcut (x & (c-1)) would be wrong
    expr = recombine_pattern("a", 6)
    _simplify_equiv(mod, expr, {"a": (0, 41)}, msg="[non-pow2 recombine]")


@case
def work_evidence_depends_on_var(mod):
    # the simplified output must still genuinely depend on the variable -- a stub
    # that returns a constant (or drops the variable) fails value-equivalence AND
    # this explicit dependence check.
    expr = recombine_pattern("i", 4)
    bounds = {"i": (0, 63)}
    out = _simplify_equiv(mod, expr, bounds, msg="[work evidence]")
    if "i" not in free_vars(out):
        raise AssertionError("simplified output dropped the variable it depends on")
    if eval_expr(out, {"i": 0}) == eval_expr(out, {"i": 1}):
        raise AssertionError("simplified output does not vary with the variable (stub?)")


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
