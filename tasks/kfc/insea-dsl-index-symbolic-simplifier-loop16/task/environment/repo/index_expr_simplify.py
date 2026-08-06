"""Canonical simplification of bounded integer index expressions.

An *index expression* is a small tree over integer constants and non-negative
bounded integer variables built from the operators ``+``, ``*``, ``//`` (floor
division), ``%`` (modulo), ``min`` and ``max``. Trees are plain nested tuples:

    ("const", n)                      an integer literal
    ("var", name)                     a variable (``name`` is a str)
    ("add",      left, right)         left + right
    ("mul",      left, right)         left * right
    ("floordiv", left, right)         left // right
    ("mod",      left, right)         left %  right
    ("min",      left, right)         min(left, right)
    ("max",      left, right)         max(left, right)

Public entry point
------------------
``simplify_expr(expr, bounds)`` -> a new expression tree.

``bounds`` maps every variable name that occurs in ``expr`` to an inclusive
integer range ``(lo, hi)`` with ``0 <= lo <= hi``. The returned tree must

  * evaluate to the SAME integer as ``expr`` for every assignment of the
    variables within their declared ranges (Python ``//`` / ``%`` semantics), and
  * contain no more nodes than ``expr`` (a simplification never grows the tree).

Error contract
--------------
``ValueError`` if a variable occurring in ``expr`` has no entry in ``bounds``
(an unbounded variable cannot be reasoned about), and ``ValueError`` if a
``floordiv`` or ``mod`` has a divisor that is the constant ``0``.

Why the current implementation leaves large expressions
------------------------------------------------------
The routine below makes a single bottom-up pass that only folds all-constant
subtrees and the most obvious per-node identities (adding ``0``, multiplying by
``1``, multiplying by ``0``). It never re-examines a node after rewriting a child
and never combines terms that are separated across ``+``, so expressions whose
size collapses only after relating a remainder to its matching quotient stay
large. Produce a tree with FEWER nodes for the same values, while keeping the
contract above exact.

Note on allowed operations
--------------------------
Build the rewrite yourself over the tuple grammar above. Do not delegate the
simplification to a computer-algebra system or any external symbolic-math package
(the scoring harness scans the submitted file for those imports and scores the
task 0). Standard library only.
"""


def _check_bounds(bounds):
    if not isinstance(bounds, dict):
        raise ValueError("bounds must be a dict {var_name: (lo, hi)}")
    for name, rng in bounds.items():
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            raise ValueError(f"bad bound for {name!r}: {rng!r}")
        lo, hi = rng
        if isinstance(lo, bool) or isinstance(hi, bool) \
                or not isinstance(lo, int) or not isinstance(hi, int):
            raise ValueError(f"bound for {name!r} must be integers: {rng!r}")
        if lo < 0 or hi < lo:
            raise ValueError(f"bound for {name!r} must satisfy 0 <= lo <= hi: {rng!r}")


def _free_vars(expr, out):
    op = expr[0]
    if op == "const":
        return
    if op == "var":
        out.add(expr[1])
        return
    _free_vars(expr[1], out)
    _free_vars(expr[2], out)


def _check_free_vars(expr, bounds):
    seen = set()
    _free_vars(expr, seen)
    missing = [v for v in seen if v not in bounds]
    if missing:
        raise ValueError(f"unbounded variable(s): {sorted(missing)}")


def _as_const(node):
    """Return the integer value if node is a constant literal, else None."""
    if node[0] == "const":
        return node[1]
    return None


def _fold(expr):
    op = expr[0]
    if op in ("const", "var"):
        return expr

    left = _fold(expr[1])
    right = _fold(expr[2])
    lc = _as_const(left)
    rc = _as_const(right)

    # all-constant subtree -> evaluate to a literal
    if lc is not None and rc is not None:
        if op == "add":
            return ("const", lc + rc)
        if op == "mul":
            return ("const", lc * rc)
        if op == "floordiv":
            if rc == 0:
                raise ValueError("division by constant zero")
            return ("const", lc // rc)
        if op == "mod":
            if rc == 0:
                raise ValueError("modulo by constant zero")
            return ("const", lc % rc)
        if op == "min":
            return ("const", lc if lc <= rc else rc)
        if op == "max":
            return ("const", lc if lc >= rc else rc)

    # a constant-zero divisor is rejected even if only one side is constant
    if op in ("floordiv", "mod") and rc == 0:
        raise ValueError("division/modulo by constant zero")

    # obvious single-node identities
    if op == "add":
        if lc == 0:
            return right
        if rc == 0:
            return left
    elif op == "mul":
        if lc == 1:
            return right
        if rc == 1:
            return left
        if lc == 0 or rc == 0:
            return ("const", 0)

    return (op, left, right)


def simplify_expr(expr, bounds):
    """See the module docstring for the full contract."""
    if bounds is None:
        bounds = {}
    _check_bounds(bounds)
    _check_free_vars(expr, bounds)
    return _fold(expr)
