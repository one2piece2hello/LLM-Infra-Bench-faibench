"""CPU instruction-count (callgrind Ir) benchmark for the caching-allocator task.

The metric is a hardware-portable deterministic proxy: the number of retired
instructions (valgrind/callgrind "I refs") for a fixed workload — building a large
pool of many distinct-size freed buffers and then replaying a long alloc/free
stream against it. The naive flat pool must LINEAR-SCAN every pooled buffer on each
alloc to make its size-exact reuse decision; a size-indexed pool makes the same
decision with an O(1) lookup, so it retires strictly fewer instructions for the
IDENTICAL observable result (decisions + device-op counts + final layout).

Invoked one MODE per process so callgrind attributes the whole-process Ir to that
mode; ``test.sh`` runs the candidate and the frozen baseline under callgrind
separately and takes the ratio baseline_Ir / candidate_Ir.

Modes:
  candidate  replay the stream using /app/repo/caching_allocator.py; print CHECKSUM=<n>
  baseline   same, using the frozen baseline module (KB_BASELINE_MODULE)
  verify     (NOT under callgrind) replay with BOTH modules and compare each to the
             independent reference; print VERIFY_OK / VERIFY_FAIL

The corpus generation is identical across modes (fixed seed/shape) so it cancels in
the ratio. Shape below is pinned; re-tune it for your hardware to land a healthy
speedup (headroom scales with pool size x rounds); a small pool flattens the win.
"""

import os
import sys

from kb_alloc_harness import (
    compare_observables,
    drive_module,
    load_candidate,
    load_module,
    make_pool_scan_stream,
    ref_drive,
)

# Long many-distinct-size pool workload (pinned; re-tune if you change hardware; a small pool or few
# rounds flattens the win because interpreter startup then dominates the ratio).
NUM_DISTINCT = 1200
ROUNDS = 8000
HIT_FRACTION = 0.5
SEED = 20260720
CAPACITY = 1_000_000_000


def _corpus():
    return make_pool_scan_stream(NUM_DISTINCT, ROUNDS, seed=SEED, hit_fraction=HIT_FRACTION)


def _checksum(obs):
    """Fold the observable result into one int so nothing is elided and modes can be
    cross-checked. Deterministic integer folding only (no salted hash)."""
    code = {"reuse": 1, "new": 2, "oom": 3}
    c = 0
    for tag in ("device_alloc_count", "device_free_count", "eviction_count", "reuse_count"):
        c = (c * 1000003 + int(obs[tag])) & 0xFFFFFFFFFFFF
    for d in obs["decisions"]:
        c = (c * 1000003 + code[d]) & 0xFFFFFFFFFFFF
    for s in obs["live_sizes"]:
        c = (c * 1000003 + s) & 0xFFFFFFFFFFFF
    for s in obs["cached_sizes"]:
        c = (c * 1000003 + s) & 0xFFFFFFFFFFFF
    return c


def _baseline_module():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/caching_allocator.py")
    return load_module(path)


def mode_candidate():
    obs = drive_module(load_candidate(), _corpus(), CAPACITY)
    print(f"CHECKSUM={_checksum(obs)}")
    return 0


def mode_baseline():
    obs = drive_module(_baseline_module(), _corpus(), CAPACITY)
    print(f"CHECKSUM={_checksum(obs)}")
    return 0


def mode_verify():
    ops = _corpus()
    ref = ref_drive(ops, CAPACITY)
    bad = 0
    for name, mod in (("candidate", load_candidate()), ("baseline", _baseline_module())):
        try:
            compare_observables(drive_module(mod, ops, CAPACITY), ref, name)
        except AssertionError as exc:
            print(f"VERIFY_FAIL {name}: {exc}")
            bad += 1
    if bad:
        return 1
    # analytic work-evidence: a flat pool must consider every pooled buffer on each
    # alloc; the size-indexed form makes the same decisions without the scan.
    print(f"VERIFY_OK ops={len(ops)} pool={NUM_DISTINCT} rounds={ROUNDS} "
          f"device_allocs={ref['device_alloc_count']} reuses={ref['reuse_count']} "
          f"flat_scan_min_iters={NUM_DISTINCT * ROUNDS}")
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
