"""Shared harness for the contiguous memory-pool defrag/coalesce task (CPU, pure Python).

Provides: candidate/baseline module loaders, an INDEPENDENT obviously-correct
reference pool (the ground truth — candidate/baseline/oracle/baseline2 are all
scored against it, never against each other), a deterministic alloc/free op-stream
generator, and the op-stream driver that folds a checksum of the observable results
(returned offsets, total-free / largest-free after each op, relocation count, and
the final live layout). Standard library only (no torch / numpy) so the metric is a
hardware-portable instruction count.
"""

import importlib.util
import os
import random

_MOD = (1 << 61) - 1


def repo_dir():
    return os.environ.get("KB_REPO_DIR", "/app/repo")


def load_candidate():
    """Import the candidate module from the repo under test."""
    path = os.path.join(repo_dir(), "mem_pool.py")
    spec = importlib.util.spec_from_file_location("candidate_mem_pool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "MemoryPool"):
        raise AttributeError(f"{path} does not define MemoryPool")
    return mod


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "kb_mem_pool_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Independent, obviously-correct reference pool (the ground truth).
#
# Live runs are held in a dict handle -> (offset, length); free runs are DERIVED
# on demand as the complement of the sorted live runs, so coalescing is automatic
# and cannot be got wrong. First-fit lowest address; compaction slides live runs to
# the front in ascending-offset order. Speed is irrelevant here — only correctness.
# --------------------------------------------------------------------------- #
def _ref_check_size(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an int")
    if size < 1:
        raise ValueError("size must be >= 1")


def _ref_check_alloc(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("allocate size must be an int")
    if size < 0:
        raise ValueError("allocate size must be >= 0")


class ReferencePool:
    def __init__(self, size):
        _ref_check_size(size)
        self.size = int(size)
        self._live = {}      # handle -> (offset, length), length > 0
        self._zero = set()   # zero-size handles
        self._next = 0
        self._reloc = 0

    def _live_sorted(self):
        return sorted(self._live.items(), key=lambda kv: kv[1][0])

    def _free_intervals(self):
        """Free runs as (start, length), ascending, derived from the live runs."""
        out = []
        cur = 0
        for _h, (off, ln) in self._live_sorted():
            if off > cur:
                out.append((cur, off - cur))
            cur = off + ln
        if cur < self.size:
            out.append((cur, self.size - cur))
        return out

    def total_free(self):
        used = sum(ln for _off, ln in self._live.values())
        return self.size - used

    def largest_free(self):
        best = 0
        for _start, ln in self._free_intervals():
            if ln > best:
                best = ln
        return best

    def relocated_blocks(self):
        return self._reloc

    def offset_of(self, handle):
        if handle in self._zero:
            return 0
        if handle not in self._live:
            raise KeyError(handle)
        return self._live[handle][0]

    def _compact(self):
        cur = 0
        for handle, (off, ln) in self._live_sorted():
            if off != cur:
                self._live[handle] = (cur, ln)
                self._reloc += 1
            cur += ln

    def allocate(self, size):
        _ref_check_alloc(size)
        if size == 0:
            handle = self._next
            self._next += 1
            self._zero.add(handle)
            return handle
        if size > self.total_free():
            return None
        offset = None
        for start, ln in self._free_intervals():  # ascending by construction
            if ln >= size:
                offset = start
                break
        if offset is None:
            self._compact()
            offset = self.size - self.total_free()  # start of the trailing free run
        handle = self._next
        self._next += 1
        self._live[handle] = (offset, size)
        return handle

    def release(self, handle):
        if handle in self._zero:
            self._zero.discard(handle)
            return
        if handle not in self._live:
            raise KeyError(handle)
        del self._live[handle]


# --------------------------------------------------------------------------- #
# Op-stream driver + checksum. Works on ANY pool object exposing the public
# contract (ReferencePool and the candidate/baseline MemoryPool alike).
#
# ops is a list of ("alloc", size) and ("free", alloc_op_index) tuples. A "free"
# refers to the allocation created by an earlier "alloc" op (by its op index); the
# driver maps that to the pool's own handle. Frees of a failed/already-freed alloc
# are skipped (a no-op), which keeps the stream valid for every pool.
# --------------------------------------------------------------------------- #
def _fold(acc, value):
    return (acc * 1000003 + (int(value) % _MOD)) % _MOD


def run_ops(pool, size, ops):
    handles = [None] * len(ops)   # pool handle per alloc-op index (None if failed)
    live_ops = set()              # alloc-op indices currently live
    checksum = 0
    for i, op in enumerate(ops):
        kind = op[0]
        if kind == "alloc":
            handle = pool.allocate(op[1])
            handles[i] = handle
            if handle is None:
                checksum = _fold(checksum, -1)
            else:
                checksum = _fold(checksum, pool.offset_of(handle))
                live_ops.add(i)
        else:  # "free"
            j = op[1]
            if j in live_ops and handles[j] is not None:
                pool.release(handles[j])
                live_ops.discard(j)
        checksum = _fold(checksum, pool.total_free())
        checksum = _fold(checksum, pool.largest_free())
    checksum = _fold(checksum, pool.relocated_blocks())
    for i in sorted(live_ops):
        checksum = _fold(checksum, pool.offset_of(handles[i]))
    return checksum


# --------------------------------------------------------------------------- #
# Deterministic op-stream generator: a long alloc/free mix over an arena with many
# small blocks -> heavy fragmentation (many free runs) + frequent releases, plus
# occasional oversized requests that force a compaction. This is the regime where a
# per-op full re-sort of the free list dominates the cost.
# --------------------------------------------------------------------------- #
def make_bench_ops(size, num_ops, seed=20260720,
                   min_block=1, max_block=48, big_block=None, free_bias=0.5,
                   big_every=64):
    if big_block is None:
        big_block = max(max_block * 3, size // 6)
    rng = random.Random(seed)
    ops = []
    alloc_indices = []  # op indices that were "alloc" (candidates to free)
    for step in range(num_ops):
        force_big = (step % big_every == big_every - 1)
        do_free = (not force_big) and alloc_indices and rng.random() < free_bias
        if do_free:
            j = rng.choice(alloc_indices)
            ops.append(("free", j))
        else:
            if force_big:
                sz = rng.randint(big_block, big_block + max_block)
            else:
                sz = rng.randint(min_block, max_block)
            ops.append(("alloc", sz))
            alloc_indices.append(step)
    return ops


def make_bench_workload(size=4096, num_ops=2400, seed=20260720):
    """The pinned metric workload (re-tune if you change hardware). Returns (size, ops)."""
    return size, make_bench_ops(size, num_ops, seed=seed)
