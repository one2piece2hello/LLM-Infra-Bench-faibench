"""Contiguous memory-pool allocator with on-demand defragmentation.

Public entry point:
    ``MemoryPool(size)`` — an allocator over a single fixed arena of ``size`` cells.
    ``MemoryPool.allocate(size)`` -> ``handle`` (an int) or ``None``
    ``MemoryPool.release(handle)`` -> ``None``
    ``MemoryPool.offset_of(handle)`` -> ``int``
    ``MemoryPool.largest_free()`` -> ``int``
    ``MemoryPool.total_free()`` -> ``int``
    ``MemoryPool.relocated_blocks()`` -> ``int``

This is the contiguous memory pool a training runtime uses to hand out many
variable-size buffers from one pre-reserved arena: allocations must be backed by a
single contiguous run of cells, freed runs are merged back with their neighbours,
and when no single free run is large enough (but the total free space is) the live
runs are compacted toward the front to open one big run. It is a pure bookkeeping
algorithm over integer offsets — no real device memory is involved.

Contract
--------
``MemoryPool(size)``: ``size`` is a positive ``int`` (the arena length in cells).
``TypeError`` if ``size`` is not an ``int`` (bools rejected); ``ValueError`` if
``size < 1``.

``allocate(size)`` -> ``handle`` | ``None``:
  * ``size`` is an ``int`` ``>= 0`` (``TypeError`` if not an int / is a bool;
    ``ValueError`` if ``size < 0``).
  * ``size == 0`` always succeeds and returns a handle that occupies no cells
    (its offset is ``0``).
  * otherwise, if ``size`` exceeds the total free space, the allocation **fails**
    and returns ``None`` (this is a normal decision, not an error).
  * otherwise the pool returns a handle backed by a contiguous run of ``size``
    cells. The run is placed at the **lowest start address** among the free runs
    large enough to hold it (first-fit, lowest address). If no single free run is
    large enough but the total free space is, the pool first **compacts** the live
    runs toward the front of the arena (relocating their contents, preserving their
    relative order) so the freed space becomes one contiguous run, then places the
    allocation there.

``release(handle)`` -> ``None``: free the run backing ``handle`` and merge it with
any immediately adjacent free run on either side (so two touching free runs never
remain separate). ``KeyError`` if the handle is unknown or already released.

``offset_of(handle)`` -> ``int``: the current base offset of a live handle
(``KeyError`` if unknown / released). Note an offset can change after a compaction.

``largest_free()`` -> ``int``: the size of the largest single contiguous free run
(``0`` if the arena is full).

``total_free()`` -> ``int``: the total number of free cells (``size`` minus the
sum of live run sizes).

``relocated_blocks()`` -> ``int``: the cumulative number of live runs whose offset
was physically changed by a compaction. It is ``> 0`` exactly when at least one
compaction has moved something, and stays ``0`` while every allocation is satisfied
without compaction.

Invariants (true after every operation)
--------------------------------------
* live runs are non-overlapping and lie within ``[0, size)``;
* ``total_free() == size - sum(live run sizes)``;
* after any ``release`` no two adjacent free runs remain un-merged;
* ``allocate(n)`` returns ``None`` iff ``n > total_free()`` (for ``n > 0``).

Why the current implementation is slow
--------------------------------------
This reference keeps the free runs in a plain list and, on **every** ``release``,
appends the freed run and then re-sorts the whole list and re-scans it to merge
neighbours; on **every** ``allocate`` it re-sorts the free list again to find the
lowest-address fit; and it recomputes ``total_free`` / ``largest_free`` by scanning
all free runs each time. When the arena is fragmented into many small free runs
(the common case after a long mix of allocations and releases), the repeated full
re-sorts and scans dominate the cost. Make the pool **faster** — do the same work
with fewer element operations for the same placements and the same decisions — for
example by keeping the free runs organized so a release merges its neighbours and
an allocation finds its fit without re-sorting the whole list every time.

Note on allowed operations
--------------------------
Implement the free-run bookkeeping, the neighbour merge, the first-fit search, and
the compaction relocation yourself. Do not delegate to a real device / OS allocator
or an array library — the scoring harness scans the submitted file for those and
scores the task 0.
"""


def _check_size(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"size must be an int, got {type(size).__name__}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")


def _check_alloc_size(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"allocate size must be an int, got {type(size).__name__}")
    if size < 0:
        raise ValueError(f"allocate size must be >= 0, got {size}")


class MemoryPool:
    """See the module docstring for the full contract.

    Naive reference allocator: free runs live in an unordered ``list`` of
    ``[start, size]`` pairs. Every ``release`` re-sorts and re-scans the whole free
    list to merge neighbours; every ``allocate`` re-sorts it again to find the
    lowest-address fit; ``total_free`` / ``largest_free`` scan all free runs. Correct,
    but the work grows with the number of free runs on every single operation.
    """

    def __init__(self, size=1):
        _check_size(size)
        self.size = int(size)
        # free runs as [start, length]; NOT kept in any order between ops.
        self._free = [[0, self.size]]
        # handle -> [offset, length] for live, non-zero-size allocations.
        self._live = {}
        # zero-size handles occupy nothing; tracked so release/offset_of work.
        self._zero = set()
        self._next_handle = 0
        self._reloc = 0

    # -- observers ---------------------------------------------------------- #
    def total_free(self):
        used = 0
        for _off, ln in self._live.values():
            used += ln
        return self.size - used

    def largest_free(self):
        best = 0
        for _start, ln in self._free:
            if ln > best:
                best = ln
        return best

    def relocated_blocks(self):
        return self._reloc

    def offset_of(self, handle):
        if handle in self._zero:
            return 0
        if handle not in self._live:
            raise KeyError(f"unknown or released handle {handle!r}")
        return self._live[handle][0]

    # -- internals ---------------------------------------------------------- #
    def _resort_free(self):
        # naive: re-sort the entire free list by start address every time.
        self._free.sort(key=lambda run: run[0])

    def _coalesce_all(self):
        # naive: after any change, re-sort and do a full linear merge pass over the
        # whole free list.
        self._resort_free()
        merged = []
        for start, ln in self._free:
            if ln <= 0:
                continue
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1][1] += ln
            else:
                merged.append([start, ln])
        self._free = merged

    def _compact(self):
        # slide every live run toward the front in ascending-offset order, keeping
        # relative order and packing with no gaps.
        order = sorted(self._live.items(), key=lambda kv: kv[1][0])
        cur = 0
        for handle, run in order:
            if run[0] != cur:
                run[0] = cur
                self._reloc += 1
            cur += run[1]
        # one trailing free run holds all the freed space.
        if cur < self.size:
            self._free = [[cur, self.size - cur]]
        else:
            self._free = []

    # -- mutators ----------------------------------------------------------- #
    def allocate(self, size):
        _check_alloc_size(size)
        if size == 0:
            handle = self._next_handle
            self._next_handle += 1
            self._zero.add(handle)
            return handle
        if size > self.total_free():
            return None
        # naive first-fit: re-sort every call, then linear scan for lowest fit.
        self._resort_free()
        chosen = -1
        for i, (_start, ln) in enumerate(self._free):
            if ln >= size:
                chosen = i
                break
        if chosen < 0:
            # no single run fits but total free does -> compact, then place at front.
            self._compact()
            chosen = 0
            for i, (_start, ln) in enumerate(self._free):
                if ln >= size:
                    chosen = i
                    break
        start, ln = self._free[chosen]
        offset = start
        if ln == size:
            self._free.pop(chosen)
        else:
            self._free[chosen] = [start + size, ln - size]
        handle = self._next_handle
        self._next_handle += 1
        self._live[handle] = [offset, size]
        return handle

    def release(self, handle):
        if handle in self._zero:
            self._zero.discard(handle)
            return
        if handle not in self._live:
            raise KeyError(f"unknown or released handle {handle!r}")
        offset, ln = self._live.pop(handle)
        # naive: append the freed run, then re-sort + full merge pass.
        self._free.append([offset, ln])
        self._coalesce_all()
