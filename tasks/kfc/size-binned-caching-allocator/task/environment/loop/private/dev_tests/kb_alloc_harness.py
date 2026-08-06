"""Shared harness for the size-exact caching-allocator task (CPU, pure Python).

Provides: a candidate loader, an INDEPENDENT obviously-correct reference driver for
the alloc/free op stream (the ground truth — candidate/baseline/oracle are all
scored against this, never against each other), deterministic op-stream generators,
and an observables comparator. Standard library only (no torch / numpy) so the
metric is a hardware-portable instruction count.

The reference uses a per-size bucket dict internally; that is just an
obviously-correct way to express the pinned reuse policy and is deliberately a
DIFFERENT body of code from the naive flat-list product baseline. Both the naive
baseline and any faster candidate must reproduce the reference's observable output
exactly (decision sequence, device-op counters, and the final live/cached size
multisets) — none of which depends on which particular same-size buffer is reused,
so free order within a size is observationally irrelevant.
"""

import importlib.util
import os
import random
import re


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def _module_name(path):
    # Deterministic module name derived from the path (NO salted hash()): a stable
    # name keeps repeated imports reproducible run-to-run.
    return "kb_alloc_mod_" + re.sub(r"[^0-9A-Za-z_]", "_", path)


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "caching_allocator.py")
    spec = importlib.util.spec_from_file_location(_module_name(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CachingAllocator"):
        raise AttributeError(f"{path} does not define CachingAllocator")
    return mod


def load_module(path):
    spec = importlib.util.spec_from_file_location(_module_name(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Op-stream encoding.
#   ("A", size)                 -> alloc(size); its handle is appended to a list
#                                  indexed by allocation order.
#   ("F", alloc_index, cache)   -> free the handle from the alloc_index-th alloc,
#                                  with cacheable=cache. Generators only emit frees
#                                  for currently-live allocs.
# Handles differ across implementations, so the stream never names a handle
# directly — it names an allocation index, which each driver resolves locally.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Independent, obviously-correct reference driver (the ground truth).
# --------------------------------------------------------------------------- #
def ref_drive(ops, capacity):
    buckets = {}          # size -> list[handle] : the reference's size-keyed pool
    size_of = {}          # handle -> size, for every device-resident buffer
    live = set()
    device_bytes = 0
    next_handle = 0
    device_alloc_count = device_free_count = eviction_count = reuse_count = 0
    handles = []
    decisions = []

    def dev_free(h):
        nonlocal device_bytes, device_free_count
        device_bytes -= size_of[h]
        del size_of[h]
        device_free_count += 1

    for op in ops:
        if op[0] == "A":
            size = op[1]
            bucket = buckets.get(size)
            if bucket:
                h = bucket.pop()
                live.add(h)
                reuse_count += 1
                decisions.append("reuse")
                handles.append(h)
                continue
            # cache miss -> a new device buffer is required
            if device_bytes + size > capacity:
                for _sz, hs in list(buckets.items()):
                    for hh in hs:
                        dev_free(hh)
                buckets = {}
                eviction_count += 1
            if device_bytes + size > capacity:
                handles.append(None)
                decisions.append("oom")
            else:
                h = next_handle
                next_handle += 1
                size_of[h] = size
                device_bytes += size
                device_alloc_count += 1
                live.add(h)
                decisions.append("new")
                handles.append(h)
        else:  # "F"
            idx, cacheable = op[1], op[2]
            h = handles[idx]
            live.discard(h)
            if cacheable:
                buckets.setdefault(size_of[h], []).append(h)
            else:
                dev_free(h)

    live_sizes = sorted(size_of[h] for h in live)
    cached_sizes = sorted(size_of[h] for hs in buckets.values() for h in hs)
    return {
        "decisions": decisions,
        "device_alloc_count": device_alloc_count,
        "device_free_count": device_free_count,
        "eviction_count": eviction_count,
        "reuse_count": reuse_count,
        "live_sizes": live_sizes,
        "cached_sizes": cached_sizes,
    }


def drive_module(module, ops, capacity):
    """Drive the candidate/baseline module's CachingAllocator through the op stream
    and collect the same observables the reference reports."""
    allocator = module.CachingAllocator(capacity)
    handles = []
    decisions = []
    for op in ops:
        if op[0] == "A":
            size = op[1]
            try:
                h = allocator.alloc(size)
                handles.append(h)
                decisions.append(allocator.decisions[-1])
            except MemoryError:
                handles.append(None)
                decisions.append("oom")
        else:  # "F"
            idx, cacheable = op[1], op[2]
            allocator.free(handles[idx], cacheable=cacheable)
    return {
        "decisions": decisions,
        "device_alloc_count": allocator.device_alloc_count,
        "device_free_count": allocator.device_free_count,
        "eviction_count": allocator.eviction_count,
        "reuse_count": allocator.reuse_count,
        "live_sizes": allocator.live_sizes(),
        "cached_sizes": allocator.cached_sizes(),
    }


def compare_observables(got, ref, msg=""):
    """Assert two observable dicts are equal, key by key, with a useful message."""
    for key in ("decisions", "device_alloc_count", "device_free_count",
                "eviction_count", "reuse_count", "live_sizes", "cached_sizes"):
        if got[key] != ref[key]:
            g, r = got[key], ref[key]
            if isinstance(g, list) and isinstance(r, list) and len(g) != len(r):
                detail = f"len {len(g)} != {len(r)}"
            else:
                detail = f"{g!r:.200} != {r!r:.200}"
            raise AssertionError(f"{key} mismatch {msg}: {detail}")


# --------------------------------------------------------------------------- #
# Deterministic op-stream generators.
# --------------------------------------------------------------------------- #
def make_churn_stream(num_ops, size_choices, seed=1234, alloc_bias=0.55,
                      noncacheable_prob=0.0):
    """A pseudo-random alloc/free churn stream over a small set of sizes. Frees only
    ever target currently-live allocs; when ``noncacheable_prob`` > 0 a fraction of
    frees release to the device instead of pooling."""
    rng = random.Random(seed)
    ops = []
    live = []          # allocation indices currently live
    next_idx = 0
    for _ in range(num_ops):
        do_alloc = (not live) or rng.random() < alloc_bias
        if do_alloc:
            ops.append(("A", rng.choice(size_choices)))
            live.append(next_idx)
            next_idx += 1
        else:
            k = rng.randrange(len(live))
            idx = live.pop(k)
            cacheable = rng.random() >= noncacheable_prob
            ops.append(("F", idx, cacheable))
    return ops


def make_pool_scan_stream(distinct_sizes, rounds, seed=99, hit_fraction=0.6):
    """Build a large distinct-size pool, then churn allocs against it — the regime
    where a flat per-buffer scan wades past many wrong-size buffers on every alloc.

    Phase 1: alloc one buffer of each size 1..distinct_sizes, then free them all
    (cacheable) so the pool holds ``distinct_sizes`` buffers of distinct sizes.
    Phase 2: for ``rounds`` iterations, either (hit) alloc+free a size that is in
    the pool, or (miss) alloc a size absent from the pool then free it
    non-cacheable so the pool size stays constant. Either way each alloc must
    consider the whole pool to make its size-exact decision.
    """
    rng = random.Random(seed)
    ops = []
    # phase 1: fill the pool
    for s in range(1, distinct_sizes + 1):
        ops.append(("A", s))
    for i in range(distinct_sizes):
        ops.append(("F", i, True))
    next_idx = distinct_sizes
    absent = distinct_sizes + 1        # a size never inserted into the pool
    for _ in range(rounds):
        if rng.random() < hit_fraction:
            size = rng.randint(1, distinct_sizes)   # in the pool -> reuse
            ops.append(("A", size))
            ops.append(("F", next_idx, True))        # return it to the pool
        else:
            ops.append(("A", absent))                # absent -> full scan + new
            ops.append(("F", next_idx, False))       # release, keep pool constant
        next_idx += 1
    return ops
