"""CPU instruction-count (callgrind Ir) benchmark for the histogram-merge task.

The metric is a hardware-portable deterministic proxy: the number of retired
instructions (valgrind/callgrind "I refs") for a fixed workload — folding many
sparse histograms into one merged histogram. The naive dense merge expands the
whole min..max index range and re-sparsifies by walking every cell; a merge that
touches only the populated buckets does strictly fewer instructions for the same
result.

Invoked one MODE per process so callgrind attributes the whole-process Ir to that
mode; ``test.sh`` runs the candidate and the frozen baseline under callgrind
separately and takes the ratio baseline_Ir / candidate_Ir.

Modes:
  candidate  build+merge using /app/repo/native_histogram_merge.py; print CHECKSUM=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (NOT under callgrind) build+merge with BOTH modules and compare each
             to the independent reference; print VERIFY_OK / VERIFY_FAIL

The measured region is build (add all) + merge; the corpus generation is identical
across modes (fixed seed/shape) so it cancels in the ratio.
"""

import os
import sys

from kb_nativehist_harness import (
    build_merged,
    canonical,
    load_candidate,
    load_module,
    make_bench_corpus,
    ref_merge,
)

# Wide-index-range workload (pinned; re-tune if you change hardware; a narrow range flattens the win).
NUM_HISTS = 64
BUCKETS_PER_HIST = 48
INDEX_SPAN = 150000        # indices drawn from [-150000, 150000) -> range up to ~300k
SCHEMA = 8
SEED = 20260720


def _corpus():
    return make_bench_corpus(
        num_hists=NUM_HISTS,
        buckets_per_hist=BUCKETS_PER_HIST,
        index_span=INDEX_SPAN,
        schema=SCHEMA,
        seed=SEED,
    )


def _checksum(hist):
    """Order-independent-safe checksum of the canonical merged histogram."""
    mask = 0xFFFFFFFFFFFF
    schema, zero, total, buckets = canonical(hist)
    cs = 0
    for v in (schema, zero, total):
        cs = (cs * 1000003 + v) & mask
    for idx, cnt in buckets:
        cs = (cs * 1000003 + idx) & mask
        cs = (cs * 1000003 + cnt) & mask
    return cs


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/native_histogram_merge.py")
    return load_module(path)


def mode_candidate():
    hists, schema = _corpus()
    merged = build_merged(load_candidate(), schema, hists)
    print(f"CHECKSUM={_checksum(merged)}")
    return 0


def mode_baseline():
    hists, schema = _corpus()
    merged = build_merged(_baseline_module(), schema, hists)
    print(f"CHECKSUM={_checksum(merged)}")
    return 0


def mode_verify():
    hists, schema = _corpus()
    ref = ref_merge(schema, hists)
    cand = build_merged(load_candidate(), schema, hists)
    base = build_merged(_baseline_module(), schema, hists)
    bad = 0
    if canonical(cand) != canonical(ref):
        print("VERIFY_FAIL candidate disagrees with reference")
        bad += 1
    if canonical(base) != canonical(ref):
        print("VERIFY_FAIL baseline disagrees with reference")
        bad += 1
    if bad:
        return 1
    # analytic work-evidence: a dense expand/re-sparsify visits the whole index
    # span; the populated buckets are far fewer.
    span = 2 * INDEX_SPAN
    populated = len(ref["buckets"])
    print(f"VERIFY_OK hists={NUM_HISTS} populated_buckets={populated} "
          f"dense_span_cells={span}")
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
