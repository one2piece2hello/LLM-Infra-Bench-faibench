"""Shared harness for the sparse exponential-bucket histogram merge task
(CPU, pure Python).

Provides: a candidate loader, an INDEPENDENT obviously-correct reference for the
merge (the ground truth — candidate/baseline/oracle are all scored against this,
never against each other), deterministic corpus generators, and a result
comparator. Standard library only (no torch / numpy) so the metric is a
hardware-portable instruction count.
"""

import importlib.util
import os
import random


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "native_histogram_merge.py")
    spec = importlib.util.spec_from_file_location(
        "candidate_native_histogram_merge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "NativeHistogramMerger"):
        raise AttributeError(f"{path} does not define NativeHistogramMerger")
    return mod


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "kb_nativehist_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Independent, obviously-correct reference (the ground truth).
# histograms: list of {"schema","zero_count","sum","buckets":[(idx,cnt),...]}.
# Returns the canonical merged histogram dict (buckets sorted, counts > 0).
# --------------------------------------------------------------------------- #
def ref_merge(schema, histograms):
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise TypeError("bad schema in reference")
    total_zero = 0
    total_sum = 0
    acc = {}
    for h in histograms:
        if h["schema"] != schema:
            raise ValueError("schema mismatch in reference")
        total_zero += int(h.get("zero_count", 0))
        total_sum += int(h.get("sum", 0))
        for idx, cnt in h.get("buckets", []):
            acc[int(idx)] = acc.get(int(idx), 0) + int(cnt)
    buckets = [(i, acc[i]) for i in sorted(acc) if acc[i] != 0]
    return {"schema": schema, "zero_count": total_zero, "sum": total_sum,
            "buckets": buckets}


def build_merged(module, schema, histograms):
    """Fold every histogram into a fresh merger and return its merged() output."""
    merger = module.NativeHistogramMerger(schema)
    for h in histograms:
        merger.add(h)
    return merger.merged()


def canonical(hist):
    """Coerce a merged-histogram return into a canonical comparable form:
    (schema, zero_count, sum, sorted tuple of (int,int) bucket pairs)."""
    if not isinstance(hist, dict):
        raise AssertionError(f"merged() must return a dict, got {type(hist)}")
    for key in ("schema", "zero_count", "sum", "buckets"):
        if key not in hist:
            raise AssertionError(f"merged() result missing key {key!r}")
    buckets = [(int(i), int(c)) for i, c in hist["buckets"]]
    idxs = [i for i, _ in buckets]
    if len(set(idxs)) != len(idxs):
        raise AssertionError("merged buckets contain a duplicate index")
    if any(c == 0 for _, c in buckets):
        raise AssertionError("merged buckets contain a zero-count entry")
    buckets_sorted = sorted(buckets, key=lambda pair: pair[0])
    return (int(hist["schema"]), int(hist["zero_count"]), int(hist["sum"]),
            tuple(buckets_sorted))


def assert_hist_equal(out, ref, msg=""):
    """out and ref are each a merged-histogram dict."""
    co = canonical(out)
    cr = canonical(ref)
    if co[0] != cr[0]:
        raise AssertionError(f"schema {co[0]} != expected {cr[0]} {msg}")
    if co[1] != cr[1]:
        raise AssertionError(f"zero_count {co[1]} != expected {cr[1]} {msg}")
    if co[2] != cr[2]:
        raise AssertionError(f"sum {co[2]} != expected {cr[2]} {msg}")
    if co[3] != cr[3]:
        raise AssertionError(
            f"per-bucket counts disagree with reference {msg}: "
            f"got {co[3][:6]}... expected {cr[3][:6]}...")


# --------------------------------------------------------------------------- #
# Deterministic corpus generators.
# --------------------------------------------------------------------------- #
def make_hist(schema, buckets, zero_count=0, extra_sum=0):
    """Build a histogram dict with sorted, positive-count buckets. ``sum`` is set
    to a deterministic aggregate (total observations) so it merges additively."""
    b = sorted(((int(i), int(c)) for i, c in buckets), key=lambda p: p[0])
    total_count = zero_count + sum(c for _, c in b) + extra_sum
    return {"schema": schema, "zero_count": int(zero_count), "sum": int(total_count),
            "buckets": b}


def make_bench_corpus(num_hists, buckets_per_hist, index_span, schema,
                      seed=20260720):
    """A wide-index-range corpus: many histograms, each with a modest number of
    populated buckets whose indices are drawn from a very wide range
    [-index_span, index_span). This is the regime where a dense expand over the
    full min..max range walks a huge mostly-empty array while the populated buckets
    are few — the shape that exposes the sparse-merge win.
    """
    rng = random.Random(seed)
    lo = -index_span
    hi = index_span
    hists = []
    for _ in range(num_hists):
        idxs = sorted(rng.sample(range(lo, hi), buckets_per_hist))
        buckets = [(i, rng.randint(1, 1000)) for i in idxs]
        zero_count = rng.randint(0, 50)
        hists.append(make_hist(schema, buckets, zero_count=zero_count))
    return hists, schema
