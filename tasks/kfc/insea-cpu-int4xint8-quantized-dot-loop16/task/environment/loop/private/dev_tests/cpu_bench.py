"""CPU instruction-count (callgrind Ir) benchmark for the block-dot task.

The metric is a hardware-portable deterministic proxy: the number of retired
instructions (valgrind/callgrind "I refs") for a fixed workload -- a batch of
block-encoded integer dot products. The naive path locates the byte for every lane
one lane at a time and multiplies each lane's integer product by the block's scale
factor separately; accumulating the integer lane products of a block first and
applying the scale factor once per block does strictly fewer instructions for the
same result.

Invoked one MODE per process so callgrind attributes the whole-process Ir to that
mode; ``test.sh`` runs the candidate and the frozen baseline under callgrind
separately and takes the ratio baseline_Ir / candidate_Ir.

Modes:
  candidate  compute the batch of dots using /app/repo/blocked_dot.py; print CHECKSUM=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (NOT under callgrind) compute with BOTH modules and compare each to the
             independent reference; print VERIFY_OK / VERIFY_FAIL

The measured region is the batch of dots; the corpus generation is identical across
modes (fixed seed/shape) and is amortized by repeating the batch, so it effectively
cancels in the ratio. The checksum folds the exact IEEE-754 bytes of every returned
value so nothing is elided and the candidate and baseline results must agree
bit-for-bit (the scale factors are dyadic and the lane products integer, so all
correct mechanisms are bit-identical).
"""

import os
import struct
import sys

from kb_blockdot_harness import (
    load_candidate,
    load_module,
    make_bench_corpus,
    ref_blocked_dot,
)

# Batch workload (pinned; re-tune if you change hardware). Several blocks per vector => the naive path
# pays the per-lane scale multiply and one-code-at-a-time unpack across every block.
# REPEATS amortizes the (mode-invariant) corpus generation so the ratio reflects the
# dot workload.
NUM_VECTORS = 6
BLOCKS_PER_VECTOR = 128
REPEATS = 8
SEED = 12345


def _corpus():
    return make_bench_corpus(
        num_vectors=NUM_VECTORS,
        blocks_per_vector=BLOCKS_PER_VECTOR,
        seed=SEED,
    )


def _fold(checksum, value):
    # bit-exact fold of a double's IEEE-754 little-endian bytes.
    for byte in struct.pack("<d", float(value)):
        checksum = (checksum * 1000003 + byte) & 0xFFFFFFFFFFFF
    return checksum


def _run_workload(module, vectors):
    """Measured region: compute the block dot of every vector, REPEATS times.
    Returns a checksum of all returned values so nothing is elided and modes can be
    cross-checked bit-for-bit."""
    checksum = 0
    for _ in range(REPEATS):
        for u_blocks, v_blocks in vectors:
            r = module.blocked_dot(u_blocks, v_blocks)
            checksum = _fold(checksum, r)
    return checksum


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/blocked_dot.py")
    return load_module(path)


def mode_candidate():
    vectors = _corpus()
    checksum = _run_workload(load_candidate(), vectors)
    print(f"CHECKSUM={checksum}")
    return 0


def mode_baseline():
    vectors = _corpus()
    checksum = _run_workload(_baseline_module(), vectors)
    print(f"CHECKSUM={checksum}")
    return 0


def mode_verify():
    vectors = _corpus()
    cand = load_candidate()
    base = _baseline_module()
    bad = 0
    for u_blocks, v_blocks in vectors:
        ref = ref_blocked_dot(u_blocks, v_blocks)
        cout = cand.blocked_dot(u_blocks, v_blocks)
        bout = base.blocked_dot(u_blocks, v_blocks)
        if not _scalar_close(cout, ref):
            print(f"VERIFY_FAIL candidate={cout!r} ref={ref!r}")
            bad += 1
        if not _scalar_close(bout, ref):
            print(f"VERIFY_FAIL baseline={bout!r} ref={ref!r}")
            bad += 1
        if bad >= 5:
            break
    if bad:
        return 1
    # analytic work-evidence: the naive path performs one scale multiply per lane;
    # accumulating per block reduces that to one per block.
    lanes = NUM_VECTORS * BLOCKS_PER_VECTOR * 32
    blocks = NUM_VECTORS * BLOCKS_PER_VECTOR
    print(f"VERIFY_OK vectors={NUM_VECTORS} blocks_per_vector={BLOCKS_PER_VECTOR} "
          f"naive_per_lane_scale_muls={lanes} per_block_scale_muls={blocks}")
    return 0


def _scalar_close(out, ref, rtol=1e-9, atol=1e-12):
    if isinstance(out, (list, tuple)):
        return False
    return abs(float(out) - float(ref)) <= (atol + rtol * abs(float(ref)))


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
