"""Correctness suite for the contiguous memory-pool defrag/coalesce contract.

Each case prints "CASE_PASS <name>" or "CASE_FAIL <name>: <reason>". The runner
counts CASE_PASS lines. Pure standard library; no GPU / torch required.

The candidate is scored against an INDEPENDENT reference (kb_mempool_harness.
ReferencePool driven through the same op stream), never against the live oracle.
"""

import sys
import traceback

from kb_mempool_harness import (
    ReferencePool,
    load_candidate,
    make_bench_ops,
    run_ops,
)

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _parity(mod, size, ops, msg=""):
    """Drive an identical op stream through the candidate and the reference; the
    folded observable traces (offsets, free sizes, relocation count, final layout)
    must be bit-identical."""
    cand = run_ops(mod.MemoryPool(size), size, ops)
    ref = run_ops(ReferencePool(size), size, ops)
    if cand != ref:
        raise AssertionError(f"trace checksum {cand} != reference {ref} {msg}")


def _live_layout(pool, handles):
    """Sorted list of (offset, size) for the given live handles."""
    return sorted((pool.offset_of(h), sz) for h, sz in handles)


def _assert_invariants(pool, live, size, msg=""):
    """live: dict handle -> size (non-zero live allocations only)."""
    runs = sorted((pool.offset_of(h), sz) for h, sz in live.items())
    prev_end = 0
    used = 0
    for off, sz in runs:
        if off < 0 or off + sz > size:
            raise AssertionError(f"run [{off},{off+sz}) out of arena [0,{size}) {msg}")
        if off < prev_end:
            raise AssertionError(f"overlap: run at {off} < prev_end {prev_end} {msg}")
        prev_end = off + sz
        used += sz
    if pool.total_free() != size - used:
        raise AssertionError(
            f"total_free {pool.total_free()} != {size - used} (size-used) {msg}")


@case
def normal_alloc_free_mix(mod):
    ops = [("alloc", 8), ("alloc", 16), ("alloc", 4), ("free", 1),
           ("alloc", 6), ("free", 0), ("alloc", 20), ("free", 2), ("alloc", 3)]
    _parity(mod, 64, ops)


@case
def normal_first_fit_lowest_address(mod):
    p = mod.MemoryPool(100)
    a = p.allocate(10)   # [0,10)
    b = p.allocate(10)   # [10,20)
    c = p.allocate(10)   # [20,30)
    p.release(b)         # hole [10,20), tail [30,100)
    d = p.allocate(5)    # lowest-address fit -> the hole at 10, NOT the tail
    if p.offset_of(d) != 10:
        raise AssertionError(f"first-fit lowest address expected offset 10, got {p.offset_of(d)}")
    if p.offset_of(a) != 0 or p.offset_of(c) != 20:
        raise AssertionError("unrelated live runs must not move without compaction")


@case
def boundary_full_arena(mod):
    p = mod.MemoryPool(20)
    a = p.allocate(20)
    if p.offset_of(a) != 0 or p.total_free() != 0 or p.largest_free() != 0:
        raise AssertionError("full-arena allocation bookkeeping wrong")
    if p.allocate(1) is not None:
        raise AssertionError("allocate into a full arena must fail (None)")
    p.release(a)
    b = p.allocate(20)
    if p.offset_of(b) != 0 or p.largest_free() != 0:
        raise AssertionError("re-allocation of the whole arena after free failed")


@case
def boundary_alloc_zero(mod):
    p = mod.MemoryPool(8)
    z = p.allocate(0)
    if p.offset_of(z) != 0 or p.total_free() != 8:
        raise AssertionError("zero-size allocation must occupy nothing at offset 0")
    a = p.allocate(8)
    if p.offset_of(a) != 0 or p.total_free() != 0:
        raise AssertionError("zero handle must not have consumed space")
    if p.offset_of(z) != 0:
        raise AssertionError("zero handle offset must stay 0")
    p.release(z)
    try:
        p.release(z)
    except KeyError:
        pass
    else:
        raise AssertionError("double release of a zero handle must raise KeyError")


@case
def degenerate_free_all_one_run(mod):
    size = 40
    p = mod.MemoryPool(size)
    hs = [p.allocate(10) for _ in range(4)]      # [0,10,20,30]
    for j in (2, 0, 3, 1):                        # scrambled free order
        p.release(hs[j])
    if p.largest_free() != size or p.total_free() != size:
        raise AssertionError(
            f"freeing all runs must coalesce to one run of {size}; got "
            f"largest={p.largest_free()} total={p.total_free()}")
    a = p.allocate(size)
    if p.offset_of(a) != 0:
        raise AssertionError("full-arena allocation after free-all must start at 0")


@case
def degenerate_full_then_reuse(mod):
    size = 16
    p = mod.MemoryPool(size)
    hs = [p.allocate(1) for _ in range(size)]
    if p.total_free() != 0 or p.largest_free() != 0:
        raise AssertionError("unit-filled arena must have zero free")
    if p.allocate(1) is not None:
        raise AssertionError("allocate into a completely full arena must fail")
    p.release(hs[8])                              # open a single 1-cell hole at 8
    r = p.allocate(1)
    if p.offset_of(r) != 8:
        raise AssertionError(f"reused hole expected offset 8, got {p.offset_of(r)}")


@case
def error_free_bad_id(mod):
    p = mod.MemoryPool(10)
    try:
        p.release(123456)
    except KeyError:
        pass
    else:
        raise AssertionError("release of an unknown handle must raise KeyError")
    a = p.allocate(4)
    p.release(a)
    try:
        p.release(a)
    except KeyError:
        pass
    else:
        raise AssertionError("double release must raise KeyError")


@case
def error_alloc_too_big_and_bad_args(mod):
    p = mod.MemoryPool(10)
    if p.allocate(11) is not None:
        raise AssertionError("allocate larger than the arena must fail with None, not raise")
    for bad, exc in ((-1, ValueError), (True, TypeError), (2.0, TypeError), ("4", TypeError)):
        try:
            p.allocate(bad)
        except exc:
            continue
        except Exception as other:  # noqa: BLE001
            raise AssertionError(f"allocate({bad!r}) raised {type(other).__name__}, expected {exc.__name__}")
        raise AssertionError(f"allocate({bad!r}) did not raise {exc.__name__}")
    for bad, exc in ((0, ValueError), (-5, ValueError), (True, TypeError), (3.0, TypeError)):
        try:
            mod.MemoryPool(bad)
        except exc:
            continue
        except Exception as other:  # noqa: BLE001
            raise AssertionError(f"MemoryPool({bad!r}) raised {type(other).__name__}, expected {exc.__name__}")
        raise AssertionError(f"MemoryPool({bad!r}) did not raise {exc.__name__}")


@case
def compaction_required(mod):
    p = mod.MemoryPool(30)
    a = p.allocate(10)   # [0,10)
    b = p.allocate(10)   # [10,20)
    c = p.allocate(10)   # [20,30)
    p.release(a)         # hole [0,10)
    p.release(c)         # hole [20,30); b still live at [10,20)
    if p.largest_free() != 10 or p.total_free() != 20:
        raise AssertionError("pre-compaction free layout wrong")
    reloc_before = p.relocated_blocks()
    d = p.allocate(15)   # no single 10-hole fits; total 20 >= 15 -> compact then place
    if d is None:
        raise AssertionError("allocation that fits only after compaction must succeed")
    if p.relocated_blocks() <= reloc_before:
        raise AssertionError("compaction must report at least one relocated block")
    if p.offset_of(b) != 0:
        raise AssertionError(f"compaction must slide b to the front; got {p.offset_of(b)}")
    if p.offset_of(d) != 10:
        raise AssertionError(f"post-compaction allocation expected offset 10, got {p.offset_of(d)}")


@case
def metamorphic_free_order_independence(mod):
    size = 40
    orders = [[0, 1, 2, 3], [1, 3, 0, 2], [3, 2, 1, 0]]
    for order in orders:
        p = mod.MemoryPool(size)
        hs = [p.allocate(10) for _ in range(4)]   # four adjacent blocks
        for j in order:
            p.release(hs[j])
        if p.largest_free() != size or p.total_free() != size:
            raise AssertionError(
                f"free order {order}: adjacent frees must fully coalesce to {size}; "
                f"got largest={p.largest_free()} total={p.total_free()}")


@case
def metamorphic_alloc_release_roundtrip(mod):
    size = 50
    p = mod.MemoryPool(size)
    keep = [p.allocate(7), p.allocate(11), p.allocate(5)]
    p.release(keep[1])                            # create a hole to make the layout nontrivial
    before_total, before_largest = p.total_free(), p.largest_free()
    before_layout = _live_layout(p, [(keep[0], 7), (keep[2], 5)])
    tmp = p.allocate(6)
    p.release(tmp)
    if (p.total_free(), p.largest_free()) != (before_total, before_largest):
        raise AssertionError("alloc-then-immediate-release must restore free totals exactly")
    if _live_layout(p, [(keep[0], 7), (keep[2], 5)]) != before_layout:
        raise AssertionError("alloc-then-immediate-release must restore the live layout exactly")


@case
def invariant_no_overlap_bookkeeping(mod):
    size = 256
    ops = make_bench_ops(size, 300, seed=7, max_block=40, big_every=40)
    p = mod.MemoryPool(size)
    handles = [None] * len(ops)
    live = {}   # handle -> size
    for i, op in enumerate(ops):
        if op[0] == "alloc":
            h = p.allocate(op[1])
            handles[i] = h
            if h is not None and op[1] > 0:
                live[h] = op[1]
        else:
            j = op[1]
            h = handles[j]
            if h is not None and h in live:
                p.release(h)
                del live[h]
        _assert_invariants(p, live, size, msg=f"after op {i} {op}")


@case
def hidden_long_op_stream(mod):
    # structurally larger / different-seed stream than the public cases (guards S7).
    size = 1024
    ops = make_bench_ops(size, 1500, seed=999983, min_block=1, max_block=64,
                         free_bias=0.45, big_every=48)
    _parity(mod, size, ops, msg="[hidden long stream]")


@case
def work_evidence_relocation_count(mod):
    # (a) a stream that never needs compaction reports zero relocations ...
    p = mod.MemoryPool(64)
    hs = [p.allocate(8) for _ in range(4)]        # [0,8,16,24), tail free
    p.release(hs[3])
    p.allocate(8)                                 # fits the tail; no compaction
    if p.relocated_blocks() != 0:
        raise AssertionError(f"no compaction should mean 0 relocations, got {p.relocated_blocks()}")
    # (b) ... while a stream that forces compaction reports a positive count.
    q = mod.MemoryPool(30)
    a = q.allocate(10)
    b = q.allocate(10)
    _c = q.allocate(10)
    q.release(a)
    q.release(_c)
    q.allocate(15)                                # forces compaction of b
    if q.relocated_blocks() <= 0:
        raise AssertionError("a compaction must report a positive relocation count")
    _ = b  # b's handle stays valid across the compaction


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
