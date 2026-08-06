"""Shared harness for the bounded integer index-expression simplifier task
(CPU, pure standard library — no torch / numpy / GPU).

Provides:
  * the candidate loader,
  * an INDEPENDENT, obviously-correct reference: an expression evaluator, a node
    counter, a free-variable collector, and a value-equivalence checker (the
    ground truth — candidate / baseline / oracle are all judged against this,
    never against each other),
  * deterministic expression-tree corpus generators.

Expression trees are plain nested tuples (a small serializable grammar):

    ("const", n)                      an integer literal ``n``
    ("var", name)                     a bounded integer variable (name is a str)
    ("add",      left, right)         left + right
    ("mul",      left, right)         left * right
    ("floordiv", left, right)         left // right   (Python floor division)
    ("mod",      left, right)         left %  right   (Python modulo)
    ("min",      left, right)         min(left, right)
    ("max",      left, right)         max(left, right)

``bounds`` is a dict ``{var_name: (lo, hi)}`` giving each variable an inclusive
integer range with 0 <= lo <= hi. The value axis is the number of tree nodes; a
correct simplifier keeps the evaluated value identical on every in-range
assignment while emitting fewer nodes.
"""

import importlib.util
import itertools
import os
import random

BINOPS = ("add", "mul", "floordiv", "mod", "min", "max")


class ExprError(Exception):
    """Raised by the reference on a structurally invalid tree or an unbound var."""


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "index_expr_simplify.py")
    spec = importlib.util.spec_from_file_location("candidate_index_expr_simplify", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "simplify_expr"):
        raise AttributeError(f"{path} does not define simplify_expr")
    return mod


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "kb_symbolic_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Independent, obviously-correct reference (the ground truth).
# --------------------------------------------------------------------------- #
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def eval_expr(expr, assign):
    """Evaluate ``expr`` under ``assign`` = {var_name: int}, with Python integer
    semantics. Raises ZeroDivisionError on //0 or %0 (defined arithmetic error);
    raises ExprError on a malformed tree or an unbound variable."""
    if not (isinstance(expr, tuple) and len(expr) >= 2):
        raise ExprError(f"not an expression node: {expr!r}")
    op = expr[0]
    if op == "const":
        if len(expr) != 2 or not _is_int(expr[1]):
            raise ExprError(f"bad const {expr!r}")
        return expr[1]
    if op == "var":
        if len(expr) != 2 or not isinstance(expr[1], str):
            raise ExprError(f"bad var {expr!r}")
        if expr[1] not in assign:
            raise ExprError(f"unbound var {expr[1]!r}")
        return assign[expr[1]]
    if op in BINOPS:
        if len(expr) != 3:
            raise ExprError(f"binary op needs 2 children: {expr!r}")
        a = eval_expr(expr[1], assign)
        b = eval_expr(expr[2], assign)
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op == "floordiv":
            return a // b            # ZeroDivisionError when b == 0
        if op == "mod":
            return a % b             # ZeroDivisionError when b == 0
        if op == "min":
            return a if a <= b else b
        return a if a >= b else b    # max
    raise ExprError(f"unknown op {op!r}")


def node_count(expr):
    """Total number of tree nodes (the value axis: fewer is better)."""
    if not (isinstance(expr, tuple) and len(expr) >= 2):
        raise ExprError(f"not an expression node: {expr!r}")
    op = expr[0]
    if op in ("const", "var"):
        return 1
    if op in BINOPS:
        if len(expr) != 3:
            raise ExprError(f"binary op needs 2 children: {expr!r}")
        return 1 + node_count(expr[1]) + node_count(expr[2])
    raise ExprError(f"unknown op {op!r}")


def free_vars(expr):
    """Set of variable names referenced anywhere in the tree."""
    if not (isinstance(expr, tuple) and len(expr) >= 2):
        raise ExprError(f"not an expression node: {expr!r}")
    op = expr[0]
    if op == "const":
        return set()
    if op == "var":
        return {expr[1]}
    if op in BINOPS:
        return free_vars(expr[1]) | free_vars(expr[2])
    raise ExprError(f"unknown op {op!r}")


def _domain_points(bounds, var_order, max_points):
    """Yield assignment tuples over the bounded domain. Exhaustive when the
    Cartesian product is small (<= max_points); otherwise a deterministic sample
    of exactly max_points draws (seeded — reproducible)."""
    ranges = [range(bounds[v][0], bounds[v][1] + 1) for v in var_order]
    total = 1
    for r in ranges:
        total *= len(r)
    if not var_order:
        yield ()
        return
    if total <= max_points:
        for pt in itertools.product(*ranges):
            yield pt
    else:
        rng = random.Random(0xC0FFEE)
        for _ in range(max_points):
            yield tuple(rng.randint(bounds[v][0], bounds[v][1]) for v in var_order)


def _outcome(expr, assign):
    """A comparable outcome for one assignment: ('val', int) or ('err', tag).
    Two expressions are equivalent at an assignment iff their outcomes match, so
    a rewrite that turns a divide-by-zero into a value (or vice versa) is caught."""
    try:
        return ("val", eval_expr(expr, assign))
    except ZeroDivisionError:
        return ("err", "zerodiv")
    except ExprError:
        return ("err", "illformed")


def values_equivalent(expr_a, expr_b, bounds, max_points=8192):
    """True iff ``expr_a`` and ``expr_b`` yield the same outcome for EVERY
    assignment of the (union of) their free variables over the bounded domain
    (exhaustive when small, else a seeded sample). Every free variable must have
    a bound; an unbounded variable makes equivalence unverifiable -> False.

    This is the semantic gate: a "simplification" that changes the value on any
    in-range assignment is rejected, regardless of how few nodes it emits."""
    fvs = sorted(free_vars(expr_a) | free_vars(expr_b))
    for v in fvs:
        if v not in bounds:
            return False
    for pt in _domain_points(bounds, fvs, max_points):
        assign = dict(zip(fvs, pt))
        if _outcome(expr_a, assign) != _outcome(expr_b, assign):
            return False
    return True


def canonical_key(expr):
    """A hashable, order-preserving key for an expression tree (used only to
    checksum bench output so nothing is elided)."""
    return repr(expr)


# --------------------------------------------------------------------------- #
# Deterministic expression-tree corpus generators.
# --------------------------------------------------------------------------- #
def C(n):
    return ("const", int(n))


def V(name):
    return ("var", name)


def recombine_pattern(var_name, modulus):
    """(v % c) + (v // c) * c  -- always equals v for integer v; the archetypal
    index-recombination idiom a simplifier collapses from 9 nodes to 1."""
    v = V(var_name)
    return ("add", ("mod", v, C(modulus)), ("mul", ("floordiv", v, C(modulus)), C(modulus)))


def make_bench_corpus(num_exprs, seed=20260720, max_var_hi=48):
    """A deterministic corpus of (expr, bounds) pairs weighted toward the
    index-recombination idiom and trivial identities (where a full rewrite engine
    removes many nodes), plus some already-minimal expressions (no headroom) so
    the workload is honest. Fixed seed/shape so it cancels across candidate and
    baseline runs."""
    rng = random.Random(seed)
    corpus = []
    moduli = [2, 3, 4, 5, 6, 8, 12, 16]
    for i in range(num_exprs):
        name = "v"
        c = moduli[rng.randrange(len(moduli))]
        hi = c * rng.randint(2, max_var_hi // c if max_var_hi // c >= 2 else 2) - 1
        bounds = {name: (0, hi)}
        kind = i % 5
        if kind == 0:
            # nested recombination wrapped in a trivial identity: ((rec)+0)*1
            expr = ("mul", ("add", recombine_pattern(name, c), C(0)), C(1))
        elif kind == 1:
            # recombination plus an extra additive constant term
            k = rng.randint(1, 7)
            expr = ("add", recombine_pattern(name, c), C(k))
        elif kind == 2:
            # (x % y) % y  with y a variable, wrapped; folds to x % y
            expr = ("mod", ("mod", V(name), V("w")), V("w"))
            bounds = {name: (0, hi), "w": (1, c)}
        elif kind == 3:
            # already-minimal: a bare sum of two variables (no reduction possible)
            expr = ("add", V(name), V("u"))
            bounds = {name: (0, hi), "u": (0, rng.randint(1, 16))}
        else:
            # constant-only subtree feeding a recombination
            expr = ("add", recombine_pattern(name, c), ("mul", C(rng.randint(1, 4)), C(0)))
        corpus.append((expr, bounds))
    return corpus
