"""Deterministic node-count benchmark for the index-expression simplifier task.

The metric is a hardware-portable, timing-free deterministic proxy per the
codegen-size family: the total number of tree nodes across the simplified output
of a fixed corpus of index expressions. A simplifier that only folds constants
and the obvious per-node identities leaves the recombination idioms intact, so it
emits strictly more nodes than one that also relates each remainder to its
matching quotient and re-runs to a fixpoint.

Invoked one MODE per process; ``test.sh`` runs the candidate and the frozen
baseline separately and takes the ratio ``baseline_nodes / candidate_nodes``.

Modes:
  candidate  simplify the corpus with /app/repo/index_expr_simplify.py; print
             NODECOUNT=<total> and CHECKSUM=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (soundness) simplify the corpus with BOTH modules and confirm each
             output is value-equivalent to the ORIGINAL over the bounded domain;
             print VERIFY_OK / VERIFY_FAIL

The corpus is identical across modes (fixed seed/shape) so it cancels in the ratio.
"""

import os
import sys

from kb_symbolic_harness import (
    canonical_key,
    load_candidate,
    load_module,
    make_bench_corpus,
    node_count,
    values_equivalent,
)

# Corpus shape (pin these; the recombination-heavy mix is where the win shows).
NUM_EXPRS = 240
SEED = 20260720


def _corpus():
    return make_bench_corpus(num_exprs=NUM_EXPRS, seed=SEED)


def _total_nodes(module, corpus):
    """Measured quantity: total node count of the simplified corpus, plus a
    checksum of the (stringified) outputs so nothing is elided."""
    total = 0
    checksum = 0
    for expr, bounds in corpus:
        out = module.simplify_expr(expr, bounds)
        total += node_count(out)
        for ch in canonical_key(out):
            checksum = (checksum * 1000003 + ord(ch)) & 0xFFFFFFFFFFFF
    return total, checksum


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/index_expr_simplify.py")
    return load_module(path)


def mode_candidate():
    total, checksum = _total_nodes(load_candidate(), _corpus())
    print(f"NODECOUNT={total}")
    print(f"CHECKSUM={checksum}")
    return 0


def mode_baseline():
    total, checksum = _total_nodes(_baseline_module(), _corpus())
    print(f"NODECOUNT={total}")
    print(f"CHECKSUM={checksum}")
    return 0


def mode_verify():
    corpus = _corpus()
    cand = load_candidate()
    base = _baseline_module()
    bad = 0
    for expr, bounds in corpus:
        c_out = cand.simplify_expr(expr, bounds)
        b_out = base.simplify_expr(expr, bounds)
        # each simplified tree must evaluate identically to the ORIGINAL for every
        # in-range assignment (the semantic gate); a wrong rewrite is caught here.
        if not values_equivalent(c_out, expr, bounds):
            print(f"VERIFY_FAIL candidate expr_head={str(expr)[:60]}")
            bad += 1
        if not values_equivalent(b_out, expr, bounds):
            print(f"VERIFY_FAIL baseline expr_head={str(expr)[:60]}")
            bad += 1
        # a simplification must never grow the tree
        if node_count(c_out) > node_count(expr):
            print(f"VERIFY_FAIL candidate grew nodes expr_head={str(expr)[:60]}")
            bad += 1
        if bad >= 5:
            break
    if bad:
        return 1
    print(f"VERIFY_OK exprs={len(corpus)}")
    return 0


MODES = {"candidate": mode_candidate, "baseline": mode_baseline, "verify": mode_verify}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    fn = MODES.get(mode)
    if fn is None:
        print(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        sys.exit(2)
    sys.exit(fn())


if __name__ == "__main__":
    main()
