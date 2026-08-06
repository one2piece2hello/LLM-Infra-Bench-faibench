"""Deterministic op-count benchmark for the graph fusion task.

The value axis is the number of ops in the OUTPUT graph after the fusion pass:
fewer ops == a better (more collapsed) equivalent graph. The metric is fully
deterministic and hardware-portable -- a structural node count, NO valgrind, NO
wall-clock. ``test.sh`` runs the candidate and the frozen baseline over the SAME
fixed corpus and takes the ratio speedup = baseline_ops / candidate_ops.

Every counted graph is first checked for EXTERNAL-OUTPUT EQUIVALENCE against the
independent evaluator (dag_harness.evaluate), so a pass that "shrinks" a graph by
dropping needed values earns no credit (it is reported non-equivalent and the
op-count is invalidated).

Modes:
  candidate  fuse the corpus with /app/repo/dag_fusion.py; print OPCOUNT=<n>
  baseline   same, with the frozen baseline module (KB_BASELINE_MODULE)
  verify     run BOTH; assert each output graph is equivalent to its input;
             print VERIFY_OK / VERIFY_FAIL
"""

import os
import random
import sys

from dag_harness import (
    evaluate,  # noqa: F401  (kept for parity / debugging)
    graphs_equivalent,
    load_candidate,
    load_module,
    make_bench_corpus,
    op_count,
)

SEED = 20260720
NUM_GRAPHS = 12
IDIOMS_PER_GRAPH = 4


def _corpus():
    return make_bench_corpus(num_graphs=NUM_GRAPHS, idioms_per_graph=IDIOMS_PER_GRAPH, seed=SEED)


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/dag_fusion.py")
    return load_module(path, "kb_baseline_dag_fusion")


def _total_opcount(module, graphs):
    """Sum output-graph op counts; also verify each output is equivalent to its
    input. Returns (total_ops, bad_count)."""
    rng = random.Random(SEED)
    total = 0
    bad = 0
    for g in graphs:
        out = module.fuse(g)
        try:
            ok, why = graphs_equivalent(g, out, rng)
        except Exception as exc:  # noqa: BLE001  (undefined tensor from an unsafe fusion)
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            bad += 1
            print(f"NONEQUIV {why}")
            if bad >= 3:
                break
            continue
        total += op_count(out)
    return total, bad


def mode_candidate():
    total, bad = _total_opcount(load_candidate(), _corpus())
    if bad:
        print("OPCOUNT=-1")
        return 1
    print(f"OPCOUNT={total}")
    return 0


def mode_baseline():
    total, bad = _total_opcount(_baseline_module(), _corpus())
    if bad:
        print("OPCOUNT=-1")
        return 1
    print(f"OPCOUNT={total}")
    return 0


def mode_verify():
    graphs = _corpus()
    ct, cb = _total_opcount(load_candidate(), graphs)
    bt, bb = _total_opcount(_baseline_module(), graphs)
    if cb or bb:
        print(f"VERIFY_FAIL candidate_bad={cb} baseline_bad={bb}")
        return 1
    raw = sum(op_count(g) for g in graphs)
    print(f"VERIFY_OK graphs={len(graphs)} input_ops={raw} baseline_ops={bt} candidate_ops={ct}")
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
