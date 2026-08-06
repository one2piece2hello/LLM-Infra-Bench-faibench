"""CPU instruction-count (callgrind Ir) benchmark for the memory-pool task.

The metric is a hardware-portable deterministic proxy: the number of retired
instructions (valgrind/callgrind "I refs") for a fixed workload — building a pool
and running a long alloc/free op stream that fragments the arena into many free
runs. The naive allocator re-sorts and re-scans the whole free list on every
release and every allocate; an allocator that keeps the free runs organized does
strictly fewer element operations for the same placements and decisions.

Invoked one MODE per process so callgrind attributes the whole-process Ir to that
mode; ``test.sh`` runs the candidate and the frozen baseline under callgrind
separately and takes the ratio baseline_Ir / candidate_Ir.

Modes:
  candidate  run the op stream using /app/repo/mem_pool.py; print CHECKSUM=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (NOT under callgrind) run the stream with the candidate AND the
             baseline and compare each to the independent reference; print
             VERIFY_OK / VERIFY_FAIL

The measured region is build + op stream; the op stream is generated identically
across modes (fixed seed/shape) so it cancels in the ratio.
"""

import os
import sys

from kb_mempool_harness import (
    ReferencePool,
    load_candidate,
    load_module,
    make_bench_workload,
    run_ops,
)

# Fragmenting workload (pinned; re-tune if you change hardware; a workload without many coexisting free
# runs flattens the win).
ARENA_SIZE = 4096
NUM_OPS = 2400
SEED = 20260720


def _workload():
    return make_bench_workload(size=ARENA_SIZE, num_ops=NUM_OPS, seed=SEED)


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/mem_pool.py")
    return load_module(path)


def mode_candidate():
    size, ops = _workload()
    checksum = run_ops(load_candidate().MemoryPool(size), size, ops)
    print(f"CHECKSUM={checksum}")
    return 0


def mode_baseline():
    size, ops = _workload()
    checksum = run_ops(_baseline_module().MemoryPool(size), size, ops)
    print(f"CHECKSUM={checksum}")
    return 0


def mode_verify():
    size, ops = _workload()
    cand = load_candidate()
    base = _baseline_module()
    ref_ck = run_ops(ReferencePool(size), size, ops)
    cand_ck = run_ops(cand.MemoryPool(size), size, ops)
    base_ck = run_ops(base.MemoryPool(size), size, ops)
    bad = 0
    if cand_ck != ref_ck:
        print(f"VERIFY_FAIL candidate checksum {cand_ck} != reference {ref_ck}")
        bad += 1
    if base_ck != ref_ck:
        print(f"VERIFY_FAIL baseline checksum {base_ck} != reference {ref_ck}")
        bad += 1
    if bad:
        return 1
    # analytic work-evidence: the number of release ops (each triggers a coalesce)
    # and the peak count of coexisting free runs the naive path re-sorts.
    num_free_ops = sum(1 for op in ops if op[0] == "free")
    print(f"VERIFY_OK ops={len(ops)} releases={num_free_ops} arena={size}")
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
