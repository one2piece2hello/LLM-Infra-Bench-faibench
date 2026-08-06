"""Correctness suite for the size-exact caching-allocator contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library; no GPU / torch required.

The candidate is scored against an INDEPENDENT reference (kb_alloc_harness.
ref_drive), never against the live oracle.
"""

import sys
import traceback

from kb_alloc_harness import (
    compare_observables,
    drive_module,
    load_candidate,
    make_churn_stream,
    make_pool_scan_stream,
    ref_drive,
)

BIG = 1_000_000_000
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _alloc(mod, capacity=BIG):
    return mod.CachingAllocator(capacity)


@case
def normal_reuse_same_size(mod):
    a = _alloc(mod)
    h1 = a.alloc(64)
    if a.decisions[-1] != "new" or a.device_alloc_count != 1:
        raise AssertionError(f"first alloc must be new: {a.decisions}, dac={a.device_alloc_count}")
    a.free(h1)
    a.alloc(64)
    if a.decisions[-1] != "reuse" or a.device_alloc_count != 1 or a.reuse_count != 1:
        raise AssertionError(
            f"same-size re-alloc must reuse: {a.decisions}, dac={a.device_alloc_count}, reuse={a.reuse_count}")


@case
def normal_distinct_sizes_no_reuse(mod):
    a = _alloc(mod)
    h8, h16 = a.alloc(8), a.alloc(16)
    if a.decisions != ["new", "new"] or a.device_alloc_count != 2:
        raise AssertionError(f"two distinct sizes -> two new: {a.decisions}")
    a.free(h8)
    a.free(h16)
    a.alloc(32)                       # distinct size -> new
    if a.decisions[-1] != "new" or a.device_alloc_count != 3:
        raise AssertionError(f"new distinct size must allocate: dac={a.device_alloc_count}")
    a.alloc(8)                        # pooled -> reuse
    if a.decisions[-1] != "reuse" or a.device_alloc_count != 3 or a.reuse_count != 1:
        raise AssertionError(f"pooled size must reuse: dac={a.device_alloc_count}, reuse={a.reuse_count}")


@case
def boundary_first_alloc_always_new(mod):
    for size in (1, 7, 1000):
        a = _alloc(mod)
        a.alloc(size)
        if a.decisions != ["new"] or a.reuse_count != 0 or a.device_alloc_count != 1:
            raise AssertionError(f"first alloc(size={size}) must be new on an empty cache")


@case
def boundary_single_size_bucket(mod):
    a = _alloc(mod)
    h = a.alloc(5)
    a.free(h)
    for _ in range(20):
        h = a.alloc(5)
        a.free(h)
    if a.device_alloc_count != 1 or a.reuse_count != 20:
        raise AssertionError(
            f"single-size churn -> 1 device alloc + 20 reuses: dac={a.device_alloc_count}, reuse={a.reuse_count}")


@case
def boundary_all_distinct_sizes(mod):
    a = _alloc(mod)
    for s in range(1, 31):
        a.alloc(s)                    # each a distinct size, none freed -> all new
    if a.device_alloc_count != 30 or a.reuse_count != 0 or a.decisions != ["new"] * 30:
        raise AssertionError(f"30 distinct live sizes -> 30 new: dac={a.device_alloc_count}")


@case
def degenerate_all_same_size_one_devalloc(mod):
    a = _alloc(mod)
    hs = [a.alloc(9) for _ in range(5)]     # 5 live of the same size -> 5 new
    if a.device_alloc_count != 5:
        raise AssertionError(f"5 concurrent same-size -> 5 new: dac={a.device_alloc_count}")
    for h in hs:
        a.free(h)
    [a.alloc(9) for _ in range(5)]          # all reuse the pool
    if a.device_alloc_count != 5 or a.reuse_count != 5:
        raise AssertionError(f"re-alloc 5 same-size must all reuse: dac={a.device_alloc_count}, reuse={a.reuse_count}")
    a.alloc(9)                              # pool now empty -> new
    if a.device_alloc_count != 6:
        raise AssertionError(f"6th concurrent same-size -> new: dac={a.device_alloc_count}")


@case
def degenerate_noncacheable_no_pool(mod):
    a = _alloc(mod)
    h = a.alloc(20)
    a.free(h, cacheable=False)              # released to device, NOT pooled
    if a.device_free_count != 1 or a.cached_sizes() != []:
        raise AssertionError(f"non-cacheable free must not pool: dfc={a.device_free_count}, cached={a.cached_sizes()}")
    a.alloc(20)                            # empty pool -> new
    if a.decisions[-1] != "new" or a.device_alloc_count != 2:
        raise AssertionError(f"after non-cacheable free the same size must allocate anew: dac={a.device_alloc_count}")


@case
def error_free_unknown_handle(mod):
    a = _alloc(mod)
    try:
        a.free(999999)
    except KeyError:
        pass
    else:
        raise AssertionError("freeing an unknown handle must raise KeyError")
    h = a.alloc(4)
    a.free(h)
    try:
        a.free(h)                          # double free -> no longer live
    except KeyError:
        pass
    else:
        raise AssertionError("double-free must raise KeyError")


@case
def error_bad_size_and_capacity(mod):
    checks = ((0, ValueError), (-3, ValueError), (True, TypeError),
              (2.0, TypeError), ("2", TypeError))
    for bad, exc in checks:
        try:
            mod.CachingAllocator(bad)
        except exc:
            pass
        except Exception as other:  # noqa: BLE001
            raise AssertionError(f"capacity={bad!r} raised {type(other).__name__}, expected {exc.__name__}")
        else:
            raise AssertionError(f"capacity={bad!r} did not raise {exc.__name__}")
    a = _alloc(mod)
    for bad, exc in checks:
        try:
            a.alloc(bad)
        except exc:
            pass
        except Exception as other:  # noqa: BLE001
            raise AssertionError(f"alloc({bad!r}) raised {type(other).__name__}, expected {exc.__name__}")
        else:
            raise AssertionError(f"alloc({bad!r}) did not raise {exc.__name__}")


@case
def error_oom_after_evict(mod):
    a = _alloc(mod, capacity=100)
    a.alloc(60)                            # live 60
    h2 = a.alloc(30)                       # live 90
    a.free(h2)                             # cached 30 (still resident, dev bytes 90)
    a.alloc(40)                            # miss: 90+40>100 -> evict the 30 -> 60, retry fits -> new
    if a.eviction_count != 1 or a.decisions[-1] != "new" or a.device_alloc_count != 3:
        raise AssertionError(
            f"alloc that fits only after eviction must evict-then-allocate: "
            f"evict={a.eviction_count}, dac={a.device_alloc_count}, last={a.decisions[-1]}")
    try:
        a.alloc(1)                         # live already 100, nothing to evict -> OOM
    except MemoryError:
        pass
    else:
        raise AssertionError("alloc that cannot fit even after eviction must raise MemoryError")


@case
def metamorphic_decisions_equal_reference(mod):
    ops = make_churn_stream(400, size_choices=[4, 8, 16, 32], seed=7, noncacheable_prob=0.2)
    got = drive_module(mod, ops, BIG)
    ref = ref_drive(ops, BIG)
    compare_observables(got, ref, "churn-stream")
    if got["reuse_count"] == 0:
        raise AssertionError("churn stream produced no reuse (workload too shallow)")


@case
def metamorphic_free_order_independence(mod):
    def run(free_order):
        a = _alloc(mod)
        hs = [a.alloc(7) for _ in range(3)]
        for k in free_order:
            a.free(hs[k])
        [a.alloc(7) for _ in range(3)]     # all reuse
        return (a.device_alloc_count, a.reuse_count, a.live_sizes(),
                a.cached_sizes(), list(a.decisions))
    r1 = run([0, 1, 2])
    r2 = run([2, 0, 1])
    if r1 != r2:
        raise AssertionError(f"free order within a size changed the observable output: {r1} != {r2}")
    if r1[0] != 3 or r1[1] != 3:
        raise AssertionError(f"expected 3 device allocs + 3 reuses, got dac={r1[0]}, reuse={r1[1]}")


@case
def hidden_interleaved_long_stream(mod):
    # (a) long many-distinct-size pool churn -> compare to the reference
    ops = make_pool_scan_stream(distinct_sizes=40, rounds=200, seed=5, hit_fraction=0.6)
    compare_observables(drive_module(mod, ops, BIG), ref_drive(ops, BIG), "pool-scan-long")
    # (b) a hand-built stream that forces two mid-stream evictions but no true OOM.
    # NOTE: the second field of an ("F", ...) op is the ALLOCATION-order index (the
    # k-th alloc), not the position in this list.
    evict_ops = [
        ("A", 20), ("A", 20), ("F", 0, True), ("F", 1, True),   # cache two 20s (dev 40)
        ("A", 30),                                              # alloc #2: miss -> evict -> new 30
        ("F", 2, True),
        ("A", 25),                                              # alloc #3: miss -> evict the 30 -> new 25
        ("F", 3, True),
    ]
    got = drive_module(mod, evict_ops, 50)
    ref = ref_drive(evict_ops, 50)
    compare_observables(got, ref, "evict-stream")
    if got["eviction_count"] != 2:
        raise AssertionError(f"evict stream expected 2 evictions, got {got['eviction_count']}")


@case
def work_evidence_devalloc_reduction(mod):
    ops = make_churn_stream(300, size_choices=[4, 8, 16], seed=3)
    got = drive_module(mod, ops, BIG)
    num_allocs = sum(1 for op in ops if op[0] == "A")
    new_count = got["decisions"].count("new")
    reuse = got["decisions"].count("reuse")
    if got["device_alloc_count"] != new_count:
        raise AssertionError(f"device_alloc_count {got['device_alloc_count']} != #new decisions {new_count}")
    if got["reuse_count"] != reuse:
        raise AssertionError(f"reuse_count {got['reuse_count']} != #reuse decisions {reuse}")
    if new_count + reuse != num_allocs:
        raise AssertionError(f"decisions ({new_count} new + {reuse} reuse) != {num_allocs} allocs")
    if reuse == 0:
        raise AssertionError("no reuse occurred -> caching did no work on this stream")
    # caching cut device allocs below the no-cache count (== num_allocs) by exactly reuse_count
    if num_allocs - got["device_alloc_count"] != reuse:
        raise AssertionError(
            f"device-alloc reduction {num_allocs - got['device_alloc_count']} != reuse_count {reuse}")
    compare_observables(got, ref_drive(ops, BIG), "work-evidence")


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
