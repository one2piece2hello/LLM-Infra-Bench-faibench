"""Correctness suite for the sparse exponential-bucket histogram merge contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library; no GPU / torch required.

The candidate is scored against an INDEPENDENT reference (kb_nativehist_harness.
ref_merge), never against the live oracle.
"""

import sys
import traceback

from kb_nativehist_harness import (
    assert_hist_equal,
    build_merged,
    canonical,
    load_candidate,
    make_hist,
    ref_merge,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _check(mod, schema, hists, msg=""):
    ref = ref_merge(schema, hists)
    out = build_merged(mod, schema, hists)
    assert_hist_equal(out, ref, msg=msg or f"[schema={schema} n={len(hists)}]")
    return canonical(out)


@case
def normal_overlapping(mod):
    # two histograms sharing indices 2 and 5 -> counts sum on the overlap
    a = make_hist(8, [(2, 3), (5, 7), (9, 1)], zero_count=4)
    b = make_hist(8, [(2, 10), (5, 1), (11, 2)], zero_count=6)
    _s, zero, _sm, buckets = _check(mod, 8, [a, b])
    if dict(buckets) != {2: 13, 5: 8, 9: 1, 11: 2}:
        raise AssertionError(f"overlapping merge wrong: {dict(buckets)}")
    if zero != 10:
        raise AssertionError(f"zero_count expected 10, got {zero}")


@case
def normal_disjoint(mod):
    # disjoint index sets -> union with counts unchanged
    a = make_hist(8, [(1, 5), (2, 5)])
    b = make_hist(8, [(100, 9), (101, 9)])
    _s, _z, _sm, buckets = _check(mod, 8, [a, b])
    if dict(buckets) != {1: 5, 2: 5, 100: 9, 101: 9}:
        raise AssertionError(f"disjoint merge wrong: {dict(buckets)}")


@case
def boundary_single_hist(mod):
    # merging exactly one histogram returns it unchanged
    a = make_hist(8, [(-3, 4), (0, 2), (7, 8)], zero_count=5)
    _s, zero, _sm, buckets = _check(mod, 8, [a])
    if dict(buckets) != {-3: 4, 0: 2, 7: 8} or zero != 5:
        raise AssertionError(f"single-hist merge wrong: {dict(buckets)} zero={zero}")


@case
def boundary_empty_and_identity(mod):
    # nothing added -> empty; and empty (+) x == x (identity element)
    _s, zero, total, buckets = _check(mod, 8, [])
    if buckets or zero != 0 or total != 0:
        raise AssertionError(f"empty merge must be empty, got {buckets} zero={zero}")
    empty = make_hist(8, [], zero_count=0)
    x = make_hist(8, [(4, 9), (8, 1)], zero_count=3)
    ref_x = ref_merge(8, [x])
    out = build_merged(mod, 8, [empty, x])
    assert_hist_equal(out, ref_x, msg="empty (+) x != x")


@case
def degenerate_all_same_bucket(mod):
    # many histograms all populating one identical index -> counts add up
    hists = [make_hist(8, [(42, k)]) for k in (1, 2, 3, 4, 5)]
    _s, _z, _sm, buckets = _check(mod, 8, hists)
    if dict(buckets) != {42: 15}:
        raise AssertionError(f"all-same-bucket merge wrong: {dict(buckets)}")


@case
def degenerate_sparse_far_apart(mod):
    # buckets at very distant indices -> both preserved (dense array would be huge)
    a = make_hist(8, [(-100000, 3)])
    b = make_hist(8, [(100000, 7)])
    _s, _z, _sm, buckets = _check(mod, 8, [a, b])
    if dict(buckets) != {-100000: 3, 100000: 7}:
        raise AssertionError(f"far-apart merge wrong: {dict(buckets)}")


@case
def error_mismatched_schema(mod):
    # folding a histogram of a different schema must be rejected
    merger = mod.NativeHistogramMerger(8)
    merger.add(make_hist(8, [(1, 1)]))
    try:
        merger.add(make_hist(6, [(1, 1)]))
    except ValueError:
        pass
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"mismatched schema raised {type(other).__name__}, expected ValueError")
    else:
        raise AssertionError("mismatched schema did not raise ValueError")


@case
def error_bad_schema_type(mod):
    # schema must be an int; bool and non-int are rejected at construction
    for bad in (True, 2.0, "8", None):
        try:
            mod.NativeHistogramMerger(bad)
        except TypeError:
            continue
        except Exception as other:  # noqa: BLE001
            raise AssertionError(
                f"schema={bad!r} raised {type(other).__name__}, expected TypeError")
        raise AssertionError(f"schema={bad!r} did not raise TypeError")


@case
def metamorphic_merge_order_independence(mod):
    # associative/commutative: any add order yields the same result
    a = make_hist(8, [(1, 2), (5, 3)], zero_count=1)
    b = make_hist(8, [(1, 4), (9, 6)], zero_count=2)
    c = make_hist(8, [(5, 5), (9, 1), (13, 7)], zero_count=3)
    ref = ref_merge(8, [a, b, c])
    for order in ([a, b, c], [c, b, a], [b, c, a], [c, a, b]):
        out = build_merged(mod, 8, order)
        assert_hist_equal(out, ref, msg=f"[order={[h['buckets'] for h in order]}]")


@case
def metamorphic_total_count_conservation(mod):
    # merged zero_count + sum(bucket counts) == same total over all inputs
    a = make_hist(8, [(2, 11), (4, 9)], zero_count=7)
    b = make_hist(8, [(2, 1), (100, 100)], zero_count=13)
    out = canonical(build_merged(mod, 8, [a, b]))
    merged_total = out[1] + sum(c for _, c in out[3])
    input_total = 0
    for h in (a, b):
        input_total += h["zero_count"] + sum(c for _, c in h["buckets"])
    if merged_total != input_total:
        raise AssertionError(
            f"total count not conserved: merged {merged_total} != inputs {input_total}")


@case
def hidden_manyhists_wide_range(mod):
    # many histograms over a wide index range -> exact merge (dense is wasteful)
    import random
    rng = random.Random(4242)
    hists = []
    for _ in range(24):
        idxs = sorted(rng.sample(range(-50000, 50000), 12))
        hists.append(make_hist(8, [(i, rng.randint(1, 100)) for i in idxs],
                               zero_count=rng.randint(0, 9)))
    _check(mod, 8, hists, msg="[wide-range many-hists]")


@case
def work_evidence_zero_and_sum(mod):
    # the zero bucket and the sum must be merged (summed), not dropped, AND the
    # per-bucket output must carry the summed COUNTS (not mere presence / 1s).
    a = make_hist(8, [(3, 40), (6, 2)], zero_count=100)
    b = make_hist(8, [(3, 60)], zero_count=25)
    _s, zero, total, buckets = _check(mod, 8, [a, b])
    if zero != 125:
        raise AssertionError(f"zero bucket dropped/wrong: got {zero}, expected 125")
    if dict(buckets).get(3) != 100:
        raise AssertionError(
            f"summed count wrong: bucket 3 = {dict(buckets).get(3)}, expected 100")
    # sum must be additive too (not zero / not one input's)
    expected_sum = a["sum"] + b["sum"]
    if total != expected_sum:
        raise AssertionError(f"sum not additive: got {total}, expected {expected_sum}")


def main():
    mod = load_candidate()
    passed = 0
    for fn_case in CASES:
        name = fn_case.__name__
        try:
            fn_case(mod)
            passed += 1
            print(f"CASE_PASS {name}")
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            print(f"CASE_FAIL {name}: {reason.splitlines()[0][:300]}")
            traceback.print_exc(file=sys.stderr)
    total = len(CASES)
    print(f"CASES_PASSED={passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
