"""Tile-size planning for a blocked matrix-multiply pipeline.

Public entry point:
    ``plan_tiling(problems, cap) -> [[Tm, Tn, Tk], ...]``

Each *problem* is one matrix multiply ``C[M,N] += A[M,K] * B[K,N]`` that runs as a
blocked loop nest over tiles of shape ``Tm x Tn`` (of C), reading ``Tm x Tk`` blocks of
A and ``Tk x Tn`` blocks of B. A block of C is kept on chip and accumulated across the
K dimension, so the off-chip data moved by one problem is::

    moved = esz * ( M*K*(N//Tn)  +  K*N*(M//Tm)  +  2*M*N )

(the A operand is streamed once for every column-block of C -> ``N//Tn`` times, the B
operand once for every row-block of C -> ``M//Tm`` times, and C is read and written
once). Larger ``Tm`` / ``Tn`` move less data; the on-chip buffer must hold one block of
each of A, B and C at once::

    footprint = (Tm*Tk + Tk*Tn + Tm*Tn) * esz   <=   cap

``plan_tiling`` returns, for every problem in order, a chosen ``[Tm, Tn, Tk]``.

Problem schema
--------------
``problems`` is a list of mappings, each with:

* ``"M"``, ``"N"``, ``"K"``: positive int dimensions.
* ``"esz"``: positive int element size in bytes.
* ``"tm_choices"``, ``"tn_choices"``, ``"tk_choices"``: non-empty lists of the allowed
  tile sizes for that axis. Every ``Tm`` choice must divide ``M`` exactly (``M % Tm ==
  0``), every ``Tn`` divide ``N``, every ``Tk`` divide ``K``.

``cap`` is the on-chip buffer capacity in bytes (a positive int), shared by all
problems.

What a valid plan must satisfy
------------------------------
For every problem the returned ``[Tm, Tn, Tk]`` must (a) be drawn from that problem's
respective choice lists, and (b) fit the capacity: ``(Tm*Tk + Tk*Tn + Tm*Tn) * esz <=
cap``. A plan that picks a tile outside the choices, or one that does not fit, is
INVALID and scores zero. (A valid choice always exists: the smallest tile of each axis
fits, because the choices always include a tile whose footprint is within ``cap``.)

Error contract
--------------
``ValueError`` if a problem is malformed: a non-positive / non-int dimension or ``esz``;
an empty choice list; a ``Tm`` choice that does not divide ``M`` (likewise Tn/Tk); a
non-positive ``cap``. ``TypeError`` for a non-int where an int is required (a ``bool``
is rejected).

Why the current implementation is wasteful
-------------------------------------------
This baseline picks the SMALLEST tile of each axis for every problem. That always fits
the buffer, but tiny tiles stream the A and B operands the maximum number of times, so
the total off-chip data moved is far larger than necessary. Choose larger tiles that
still fit the buffer so each operand is streamed fewer times, while keeping every
choice valid (from the lists, and within ``cap``).
"""


def _check_pos_int(v, name):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be an int, got {type(v).__name__}")
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _validate(problems, cap):
    if not isinstance(problems, list) or len(problems) == 0:
        raise ValueError("problems must be a non-empty list")
    _check_pos_int(cap, "cap")
    norm = []
    for pi, p in enumerate(problems):
        if not isinstance(p, dict):
            raise TypeError(f"problem {pi} must be a mapping")
        M = _check_pos_int(p.get("M"), f"problem {pi} M")
        N = _check_pos_int(p.get("N"), f"problem {pi} N")
        K = _check_pos_int(p.get("K"), f"problem {pi} K")
        esz = _check_pos_int(p.get("esz"), f"problem {pi} esz")
        axes = {}
        for name, dim in (("tm_choices", M), ("tn_choices", N), ("tk_choices", K)):
            choices = p.get(name)
            if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                raise ValueError(f"problem {pi} {name} must be a non-empty list")
            out = []
            for c in choices:
                if isinstance(c, bool) or not isinstance(c, int):
                    raise TypeError(f"problem {pi} {name} entry must be an int")
                if c <= 0 or dim % c != 0:
                    raise ValueError(f"problem {pi} {name} entry {c} must be a positive divisor of {dim}")
                out.append(c)
            axes[name] = out
        norm.append((M, N, K, esz, axes["tm_choices"], axes["tn_choices"], axes["tk_choices"]))
    return norm


def plan_tiling(problems, cap):
    """Frozen baseline (candidate start state).

    Picks the smallest tile of every axis for every problem: always valid (smallest
    footprint fits ``cap``) but moves the most off-chip data because the operands are
    streamed the maximum number of times. Return larger, still-fitting tiles instead.
    """
    norm = _validate(problems, cap)
    plan = []
    for (M, N, K, esz, tm, tn, tk) in norm:
        plan.append([min(tm), min(tn), min(tk)])  # smallest tiles -> maximal traffic
    return plan
