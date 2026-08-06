"""Size-exact caching allocator over a simulated fixed-capacity device.

Public entry points:
    ``CachingAllocator(capacity)`` — a buffer pool over one simulated device.
    ``CachingAllocator.alloc(size) -> handle`` — hand out a buffer of ``size``
        cells, reusing a cached freed buffer of the SAME size if one is available,
        otherwise creating a new one on the device.
    ``CachingAllocator.free(handle, cacheable=True)`` — return a buffer to the pool
        (kept for reuse) or, when ``cacheable=False``, release it to the device.

This is the caching-allocator pattern used by a training runtime to avoid repeated
device ``malloc``/``free``: freed buffers are pooled and reused by size, so a
free-then-alloc of the same size performs NO new device allocation. The "device"
here is a stub whose ``malloc``/``free`` only bump counters and track a byte total
(the real cost we are modelling), so the whole thing is a pure-Python bookkeeping
algorithm.

Contract
--------
``capacity``: a positive integer — the hard byte capacity of the simulated device
(bytes physically resident: live buffers PLUS pooled/cached buffers). ``TypeError``
if not an ``int`` (bools rejected); ``ValueError`` if ``< 1``.

``alloc(size) -> handle``: ``size`` is a positive ``int`` (``TypeError`` if not an
int / bools rejected; ``ValueError`` if ``< 1``). Behaviour:
  * If a cached (freed) buffer of EXACTLY ``size`` is available, reuse it — no new
    device allocation. This records the decision ``"reuse"``.
  * Otherwise a new buffer must be created on the device. If it would exceed
    ``capacity``, first evict the ENTIRE cache (return every cached buffer to the
    device, reclaiming its bytes) and retry once. If it still does not fit, raise
    ``MemoryError``. Otherwise create it. This records the decision ``"new"``.
The return value is an opaque integer ``handle`` identifying the buffer.

``free(handle, cacheable=True)``: ``handle`` must be a currently-live buffer
(``KeyError`` otherwise — covers freeing an unknown handle and double-free). When
``cacheable`` is true the buffer is returned to the pool for its size (still
resident on the device, reusable by a later same-size ``alloc``); when false it is
released to the device immediately (its bytes are reclaimed and it is not pooled).

Observable state (the verifier reads these; do NOT rename them or change their
meaning):
  * ``decisions`` — a list, one entry per ``alloc``, each ``"reuse"`` or ``"new"``.
  * ``device_alloc_count`` — number of NEW device buffers created (the expensive op).
  * ``device_free_count`` — number of buffers released back to the device.
  * ``eviction_count`` — number of whole-cache evictions performed (on capacity miss).
  * ``reuse_count`` — number of allocs served from the cache.
  * ``live_sizes()`` — sorted list of the sizes of currently-live buffers.
  * ``cached_sizes()`` — sorted list of the sizes of currently-pooled buffers.

Invariants: reuse is SIZE-EXACT (a request never reuses a different-size buffer); a
freed-then-alloc'd same-size request creates ZERO new device buffers; the bytes
resident on the device (live + cached) never exceed ``capacity``; a whole-cache
eviction happens only when a new allocation would otherwise exceed ``capacity``.

Why the current implementation is slow
--------------------------------------
Every freed buffer is dropped into a SINGLE flat list. To serve ``alloc(size)`` the
code LINEAR-SCANS that whole list looking for a buffer whose size equals the
request. When many buffers of many different sizes are pooled at once, each alloc
walks past a long run of wrong-size buffers before it finds a match (or gives up
and allocates), so the lookup cost grows with the number of pooled buffers. Make
the lookup faster — do fewer per-buffer comparisons for the same decisions — for
example by organizing the pooled buffers so a request of a given size goes straight
to the buffers of that size instead of scanning every pooled buffer.

Note on allowed operations
--------------------------
Implement the pooling, the size-exact reuse decision, and the evict-and-retry
control flow yourself. Do not delegate to a real memory allocator or import an
array / tensor library or the upstream device framework to do the bookkeeping — the
scoring harness scans the submitted file for those and scores the task 0.
"""


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


class CachingAllocator:
    """See the module docstring for the full contract.

    Naive reference container: pooled buffers live in one flat list and every
    ``alloc`` scans the entire list for a size-exact match. Correct, but the lookup
    work grows linearly with the number of pooled buffers.
    """

    def __init__(self, capacity=1_000_000_000):
        _check_positive_int("capacity", capacity)
        self.capacity = int(capacity)
        # device model (a stub: the "device" ops are just counters + a byte total)
        self._device_bytes = 0        # bytes resident on the device (live + cached)
        self._next_handle = 0
        self._size_of = {}            # handle -> size, for every device-resident buffer
        self._live = set()            # handles currently handed out
        # NAIVE pool: a single flat list of cached (freed) buffers as (handle, size)
        # pairs. alloc LINEAR-SCANS this whole list to find a size-exact match.
        self._free = []               # list[(handle, size)]
        # observable counters / decision log
        self.device_alloc_count = 0
        self.device_free_count = 0
        self.eviction_count = 0
        self.reuse_count = 0
        self.decisions = []

    # --- device stub: the "expensive" operations we are modelling ---
    def _device_alloc(self, size):
        handle = self._next_handle
        self._next_handle += 1
        self._size_of[handle] = size
        self._device_bytes += size
        self.device_alloc_count += 1
        return handle

    def _device_free(self, handle):
        self._device_bytes -= self._size_of[handle]
        del self._size_of[handle]
        self.device_free_count += 1

    def _evict_cache(self):
        # Return EVERY cached buffer to the device (reclaiming its bytes) and empty
        # the pool. Triggered only when a new allocation would exceed capacity.
        for handle, _size in self._free:
            self._device_free(handle)
        self._free = []
        self.eviction_count += 1

    def _find_cached(self, size):
        # NAIVE lookup: scan the whole flat pool for a size-exact match.
        for i in range(len(self._free)):
            handle, bsize = self._free[i]
            if bsize == size:
                del self._free[i]
                return handle
        return None

    def alloc(self, size):
        _check_positive_int("size", size)
        size = int(size)
        handle = self._find_cached(size)
        if handle is not None:
            self._live.add(handle)
            self.reuse_count += 1
            self.decisions.append("reuse")
            return handle
        # cache miss -> a new device buffer is required
        if self._device_bytes + size > self.capacity:
            self._evict_cache()
        if self._device_bytes + size > self.capacity:
            raise MemoryError(
                f"out of device capacity: need {size}, "
                f"{self.capacity - self._device_bytes} free")
        handle = self._device_alloc(size)
        self._live.add(handle)
        self.decisions.append("new")
        return handle

    def free(self, handle, cacheable=True):
        if handle not in self._live:
            raise KeyError(f"handle {handle!r} is not a live buffer")
        self._live.discard(handle)
        if cacheable:
            self._free.append((handle, self._size_of[handle]))
        else:
            self._device_free(handle)

    def live_sizes(self):
        return sorted(self._size_of[h] for h in self._live)

    def cached_sizes(self):
        return sorted(size for _handle, size in self._free)
