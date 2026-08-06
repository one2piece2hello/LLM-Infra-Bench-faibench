"""Correctness suite for the graph normalize + idiom-collapse fusion contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>"; the runner
counts CASE_PASS lines. Pure standard library.

The candidate is scored against an INDEPENDENT evaluator (dag_harness.evaluate):
its output graph must evaluate to the SAME external outputs as the input graph on
random inputs. Correctness never requires matching the oracle's graph. A pass that
cannot safely collapse a subgraph must leave it expanded (still equivalent).
"""

import random
import sys
import traceback

from dag_harness import (
    add_identity_noise,
    build_idiom,
    evaluate,
    graphs_equivalent,
    load_candidate,
    new_graph,
    op_count,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _rng():
    return random.Random(4242)


def _equiv(mod, g, msg=""):
    out = mod.fuse(g)
    ok, why = graphs_equivalent(g, out, _rng())
    if not ok:
        raise AssertionError(f"output not equivalent to input: {why} {msg}")
    return out


def _reduces(mod, g, msg=""):
    before = op_count(g)
    out = _equiv(mod, g, msg)
    if op_count(out) >= before:
        raise AssertionError(f"expected fewer ops (had {before}, got {op_count(out)}) {msg}")
    return out


@case
def normal_canonical_one_idiom(mod):
    g = new_graph()
    g["inputs"].append("x")
    y = build_idiom(g, "n0", "x", form="canonical")
    g["outputs"].append(y)
    # Correctness = external-output equivalence only. Op-count reduction is the
    # BENCHMARK axis (segcount corpus), NOT a correctness gate: the frozen naive
    # baseline must pass its own suite (no-op ties speedup 1.0), and it need not
    # collapse a standalone idiom whose output is a graph output.
    _equiv(mod, g, "[canonical]")


@case
def normal_two_independent_idioms(mod):
    g = new_graph()
    for k in range(2):
        g["inputs"].append(f"x{k}")
        g["outputs"].append(build_idiom(g, f"m{k}", f"x{k}", form="canonical"))
    _equiv(mod, g, "[two idioms]")


@case
def boundary_idiom_feeds_trailing_op(mod):
    # idiom output is consumed by a trailing op before the external output
    g = new_graph()
    g["inputs"].append("x")
    y = build_idiom(g, "b0", "x", form="canonical")
    g["constants"]["one"] = [1.0]
    g["nodes"].append({"op": "Mul", "name": "tail", "inputs": [y, "one"],
                       "outputs": ["z"], "attrs": {}})
    g["outputs"].append("z")
    _equiv(mod, g, "[trailing op]")


@case
def boundary_dupsub_variant_stays_equivalent(mod):
    # duplicated-Sub variant: fusing is optional, equivalence is mandatory
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append(build_idiom(g, "d0", "x", form="dupsub"))
    _equiv(mod, g, "[dupsub]")


@case
def degenerate_no_idiom_unchanged(mod):
    # plain compute, no idiom, no pass-through -> must be returned unchanged
    g = new_graph()
    g["inputs"] += ["a", "b"]
    g["constants"]["k"] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    g["nodes"].append({"op": "Mul", "name": "p0", "inputs": ["a", "k"], "outputs": ["t0"], "attrs": {}})
    g["nodes"].append({"op": "Add", "name": "p1", "inputs": ["t0", "b"], "outputs": ["t1"], "attrs": {}})
    g["outputs"].append("t1")
    before = op_count(g)
    out = _equiv(mod, g, "[no idiom]")
    if op_count(out) != before:
        raise AssertionError(f"no-idiom graph must be unchanged: {before} -> {op_count(out)}")


@case
def degenerate_output_is_input(mod):
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append("x")
    out = _equiv(mod, g, "[passthrough]")
    if op_count(out) != 0:
        raise AssertionError(f"expected empty node list, got {op_count(out)}")


@case
def error_bad_exponent_not_fused(mod):
    # exponent != 2 -> collapsing into the square-based fused node would change
    # the result, so it must NOT be collapsed (equivalence would break). Use an
    # even exponent so the reference chain itself stays evaluable (mean of even
    # powers is non-negative -> Sqrt is defined).
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append(build_idiom(g, "e0", "x", form="canonical", exponent=4.0))
    _equiv(mod, g, "[exponent 4]")


@case
def error_nonconstant_epsilon_not_fused(mod):
    g = new_graph()
    g["inputs"].append("x")
    y = build_idiom(g, "e1", "x", form="canonical")
    # turn the epsilon constant into a runtime input (not representable as an attr)
    g["constants"].pop("e1_eps")
    g["inputs"].append("e1_eps")
    g["outputs"].append(y)
    _equiv(mod, g, "[runtime epsilon]")


@case
def error_interior_escape_not_fused(mod):
    # an interior tensor (std) is consumed outside the idiom -> collapsing would
    # delete a value the second output still needs.
    g = new_graph()
    g["inputs"].append("x")
    y = build_idiom(g, "s0", "x", form="canonical")
    g["constants"]["two2"] = [2.0]
    g["nodes"].append({"op": "Pow", "name": "leak", "inputs": ["s0_std", "two2"],
                       "outputs": ["leaked"], "attrs": {}})
    g["outputs"] += [y, "leaked"]
    _equiv(mod, g, "[interior escape]")


@case
def metamorphic_node_reorder_same_count(mod):
    g = new_graph()
    for k in range(2):
        g["inputs"].append(f"x{k}")
        g["outputs"].append(build_idiom(g, f"r{k}", f"x{k}", form="canonical"))
    n1 = op_count(mod.fuse(g))
    g2 = {"inputs": list(g["inputs"]), "outputs": list(g["outputs"]),
          "constants": dict(g["constants"]), "nodes": list(reversed(g["nodes"]))}
    n2 = op_count(mod.fuse(g2))
    if n1 != n2:
        raise AssertionError(f"node order changed the op count: {n1} vs {n2}")
    _equiv(mod, g2, "[reordered]")


@case
def metamorphic_unrelated_branch(mod):
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append(build_idiom(g, "u0", "x", form="canonical"))
    g["inputs"].append("w")
    g["constants"]["c3"] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    g["nodes"].append({"op": "Add", "name": "un", "inputs": ["w", "c3"], "outputs": ["wo"], "attrs": {}})
    g["outputs"].append("wo")
    _equiv(mod, g, "[unrelated branch]")


@case
def hidden_cast_variant_stays_equivalent(mod):
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append(build_idiom(g, "h0", "x", form="cast"))
    _equiv(mod, g, "[cast variant]")


@case
def hidden_shared_input_two_idioms(mod):
    g = new_graph()
    g["inputs"].append("x")
    g["outputs"].append(build_idiom(g, "sh0", "x", form="canonical"))
    g["outputs"].append(build_idiom(g, "sh1", "x", form="cast"))
    _equiv(mod, g, "[shared input]")


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
