"""Layer-wise weight prefetch over a pinned staging ring.

Running a model whose weights do not fit in device memory means streaming them in
while the previous layers compute. The scheduler that does this has to answer, in
this order:

1. how each layer's weight bytes split into fixed-size **transfer chunks**;
2. how far ahead of the layer being computed the prefetcher is allowed to run --
   the **lookahead window**, bounded both by a layer count and by the number of
   staging slots;
3. which **pinned ring slot** each chunk lands in, which chunk it evicts, and how
   often each slot is reused;
4. when each chunk **arrives**, given that all transfers share one link with a
   fixed per-transfer overhead;
5. the per-layer **stall profile**: when a layer can start, when it ends, and how
   long compute sat idle waiting for weights;
6. the whole **pipeline report**, including the slots that were overwritten before
   their previous occupant had been consumed.

Conventions that apply to the whole module:

* Every array is ``int64`` and every division is **integer** division. ``ceil`` is
  ``(a + b - 1) // b`` on non-negative integers. Times are microseconds, sizes are
  bytes; both are non-negative.
* Chunks are numbered globally in **ascending (layer, offset-within-layer)** order,
  which is also the order they are issued on the link.
* A layer with zero weight bytes contributes zero chunks. It is legal, and it is
  ready immediately.
* No function may modify any of its inputs.
"""

import numpy as np
def layer_chunk_table(layer_bytes, chunk_bytes):
    """Split every layer's weight bytes into transfer chunks.

    Layer ``l`` of ``layer_bytes`` contributes
    ``ceil(layer_bytes[l] / chunk_bytes)`` chunks, all of size ``chunk_bytes``
    except the last one, which carries the remainder. Chunks are numbered globally
    in ascending ``(layer, offset)`` order.

    Returns ``(chunk_layer, chunk_size, layer_ptr)``: ``int64 (C,)`` owning layer
    per chunk, ``int64 (C,)`` chunk sizes, and ``int64 (L + 1,)`` prefix offsets so
    that layer ``l`` owns chunks ``layer_ptr[l] : layer_ptr[l + 1]``. An empty
    ``layer_bytes`` yields two empty arrays and ``layer_ptr == [0]``. Raises
    ``ValueError`` if ``layer_bytes`` is not 1-D, if any entry is negative, or if
    ``chunk_bytes`` is not positive.
    """
    lb = np.asarray(layer_bytes, dtype=np.int64)
    cb = int(chunk_bytes)
    if lb.ndim != 1:
        raise ValueError("layer_bytes must be 1-D")
    if cb <= 0:
        raise ValueError("chunk_bytes must be positive")
    if lb.size and int(lb.min()) < 0:
        raise ValueError("layer_bytes must be non-negative")
    # SLOW-BUT-CORRECT reference path: one python step per chunk, plus a scalar numpy read per layer.
    L = int(lb.shape[0])
    c_layer = []
    c_size = []
    ptr = [0]
    for l in range(L):
        rem = int(lb[l])
        while rem > 0:
            take = cb if rem > cb else rem
            c_layer.append(l)
            c_size.append(take)
            rem -= take
        ptr.append(len(c_layer))
    return (np.asarray(c_layer, dtype=np.int64).reshape(-1),
            np.asarray(c_size, dtype=np.int64).reshape(-1),
            np.asarray(ptr, dtype=np.int64))

def prefetch_window(layer_ptr, lookahead, ring_slots):
    """Exclusive chunk bound the prefetcher may reach while layer ``l`` computes.

    While layer ``l`` is being computed the prefetcher may have fetched any chunk
    below ``window[l] = min(layer_ptr[min(l + 1 + lookahead, L)],
    layer_ptr[l] + ring_slots)`` -- it may run ``lookahead`` whole layers past the
    layer that follows the current one, but never more than ``ring_slots`` chunks
    past the first chunk of the current layer, because that is all the staging
    space there is.

    Returns ``int64 (L,)``; ``L == 0`` yields an empty array. Raises ``ValueError``
    if ``layer_ptr`` is not 1-D, is empty, is not non-decreasing, does not start at
    zero, if ``lookahead`` is negative, or if ``ring_slots`` is not positive.
    """
    ptr = np.asarray(layer_ptr, dtype=np.int64)
    la = int(lookahead)
    rs = int(ring_slots)
    if ptr.ndim != 1 or ptr.shape[0] == 0:
        raise ValueError("layer_ptr must be a non-empty 1-D array")
    if int(ptr[0]) != 0:
        raise ValueError("layer_ptr must start at zero")
    if ptr.shape[0] > 1 and int(np.diff(ptr).min()) < 0:
        raise ValueError("layer_ptr must be non-decreasing")
    if la < 0:
        raise ValueError("lookahead must be non-negative")
    if rs <= 0:
        raise ValueError("ring_slots must be positive")
    # SLOW-BUT-CORRECT reference path: one python step per layer with scalar numpy indexing.
    L = int(ptr.shape[0]) - 1
    out = np.zeros(L, dtype=np.int64)
    for l in range(L):
        far = l + 1 + la
        if far > L:
            far = L
        a = int(ptr[far])
        b = int(ptr[l]) + rs
        out[l] = a if a < b else b
    return out

def ring_assign(chunk_size, ring_slots, slot_bytes):
    """Assign chunks to pinned ring slots, round-robin.

    Chunk ``i`` lands in slot ``i % ring_slots`` and evicts chunk
    ``i - ring_slots`` (``-1`` for the first pass over the ring).

    Returns ``(slot, evicted, reuse)``: ``int64 (C,)``, ``int64 (C,)`` and
    ``int64 (ring_slots,)`` where ``reuse[s]`` counts the chunks assigned to slot
    ``s``. Raises ``ValueError`` if ``chunk_size`` is not 1-D, if ``ring_slots`` or
    ``slot_bytes`` is not positive, or if any chunk is larger than ``slot_bytes``.
    """
    cs = np.asarray(chunk_size, dtype=np.int64)
    rs = int(ring_slots)
    sb = int(slot_bytes)
    if cs.ndim != 1:
        raise ValueError("chunk_size must be 1-D")
    if rs <= 0:
        raise ValueError("ring_slots must be positive")
    if sb <= 0:
        raise ValueError("slot_bytes must be positive")
    if cs.size and int(cs.max()) > sb:
        raise ValueError("a chunk does not fit in a staging slot")
    # SLOW-BUT-CORRECT reference path: one python step per chunk.
    C = int(cs.shape[0])
    slot = np.zeros(C, dtype=np.int64)
    ev = np.zeros(C, dtype=np.int64)
    reuse = np.zeros(rs, dtype=np.int64)
    for i in range(C):
        s = i % rs
        slot[i] = s
        ev[i] = i - rs if i >= rs else -1
        reuse[s] += 1
    return slot, ev, reuse

def chunk_arrivals(chunk_size, bw_bytes_per_us, fixed_us, issue_us):
    """When each chunk finishes landing in its staging slot.

    All chunks share one link and are issued back to back in chunk order. Chunk
    ``i`` costs ``ceil(chunk_size[i] / bw_bytes_per_us) + fixed_us`` microseconds,
    the first one starts at ``issue_us``, and each subsequent one starts when its
    predecessor finished.

    Returns ``(start, done)``, both ``int64 (C,)``. An empty input yields two empty
    arrays. Raises ``ValueError`` if ``chunk_size`` is not 1-D, if any entry is
    negative, if ``bw_bytes_per_us`` is not positive, or if ``fixed_us`` or
    ``issue_us`` is negative.
    """
    cs = np.asarray(chunk_size, dtype=np.int64)
    bw = int(bw_bytes_per_us)
    fx = int(fixed_us)
    t0 = int(issue_us)
    if cs.ndim != 1:
        raise ValueError("chunk_size must be 1-D")
    if cs.size and int(cs.min()) < 0:
        raise ValueError("chunk_size must be non-negative")
    if bw <= 0:
        raise ValueError("bw_bytes_per_us must be positive")
    if fx < 0:
        raise ValueError("fixed_us must be non-negative")
    if t0 < 0:
        raise ValueError("issue_us must be non-negative")
    # SLOW-BUT-CORRECT reference path: a python loop that walks the link queue one transfer at a time.
    C = int(cs.shape[0])
    start = np.zeros(C, dtype=np.int64)
    done = np.zeros(C, dtype=np.int64)
    t = t0
    for i in range(C):
        cost = (int(cs[i]) + bw - 1) // bw + fx
        start[i] = t
        t += cost
        done[i] = t
    return start, done

def stall_profile(layer_ptr, chunk_done, compute_us):
    """When each layer runs, and how long compute waited for its weights.

    Layer ``l`` is ready at ``ready[l] = max(chunk_done[layer_ptr[l] :
    layer_ptr[l + 1]])``, or ``0`` when the layer owns no chunks. Layers run in
    order on one device, so ``start[l] = max(ready[l], end[l - 1])`` with
    ``end[-1] = 0``, ``end[l] = start[l] + compute_us[l]`` and
    ``stall[l] = start[l] - end[l - 1]``.

    Returns ``(start, end, stall)``, all ``int64 (L,)``. Raises ``ValueError`` if
    ``layer_ptr`` is not 1-D or empty, if ``compute_us`` is not 1-D of length
    ``len(layer_ptr) - 1``, if any compute time is negative, or if ``layer_ptr``
    does not index ``chunk_done`` exactly (``layer_ptr[0] == 0`` and
    ``layer_ptr[-1] == len(chunk_done)``).
    """
    ptr = np.asarray(layer_ptr, dtype=np.int64)
    cd = np.asarray(chunk_done, dtype=np.int64)
    cu = np.asarray(compute_us, dtype=np.int64)
    if ptr.ndim != 1 or ptr.shape[0] == 0:
        raise ValueError("layer_ptr must be a non-empty 1-D array")
    L = int(ptr.shape[0]) - 1
    if cu.ndim != 1 or int(cu.shape[0]) != L:
        raise ValueError("compute_us must be 1-D of length len(layer_ptr) - 1")
    if cu.size and int(cu.min()) < 0:
        raise ValueError("compute_us must be non-negative")
    if cd.ndim != 1:
        raise ValueError("chunk_done must be 1-D")
    if int(ptr[0]) != 0 or int(ptr[L]) != int(cd.shape[0]):
        raise ValueError("layer_ptr does not index chunk_done")
    # SLOW-BUT-CORRECT reference path: a python loop per layer that re-scans the layer's chunk arrivals.
    start = np.zeros(L, dtype=np.int64)
    end = np.zeros(L, dtype=np.int64)
    stall = np.zeros(L, dtype=np.int64)
    prev = 0
    for l in range(L):
        lo = int(ptr[l])
        hi = int(ptr[l + 1])
        ready = 0
        for j in range(lo, hi):
            v = int(cd[j])
            if v > ready:
                ready = v
        s = ready if ready > prev else prev
        start[l] = s
        stall[l] = s - prev
        prev = s + int(cu[l])
        end[l] = prev
    return start, end, stall

def run_pipeline(layer_bytes, chunk_bytes, ring_slots, slot_bytes, lookahead,
                 bw_bytes_per_us, fixed_us, compute_us, issue_us=0):
    """Plan and score a whole layer-wise weight-prefetch pipeline.

    Chunks the layers, assigns the ring, computes the arrivals and the stall
    profile, and then reports the two things a capacity planner asks for: how much
    of the wall time was a **bubble** (compute idle, waiting on the link), and how
    many ring slots were **overwritten too early** -- a chunk whose transfer starts
    before the chunk it evicts has been consumed. A chunk is consumed when the
    layer that owns it finishes computing, i.e. at ``end[chunk_layer[i]]``.

    Returns a dict with the python ints ``n_chunks``, ``n_layers``,
    ``bytes_total``, ``total_us`` (``end[-1]``, or ``0`` with no layers),
    ``stall_us`` (the sum of the per-layer stalls), ``bubbles`` (how many layers
    stalled at all), ``conflicts``, ``link_busy_us`` (the sum of the per-chunk
    transfer costs) and ``last_arrival_us``; plus the arrays ``window``
    (``int64 (L,)``), ``layer_stall`` (``int64 (L,)``), ``layer_start``
    (``int64 (L,)``), ``slot_reuse`` (``int64 (ring_slots,)``) and
    ``conflict_chunks`` (``int64``, the chunk indices whose eviction was early, in
    ascending order).

    Raises ``ValueError`` whenever any of the callees would.
    """
    c_layer, c_size, ptr = layer_chunk_table(layer_bytes, chunk_bytes)
    window = prefetch_window(ptr, lookahead, ring_slots)
    slot, ev, reuse = ring_assign(c_size, ring_slots, slot_bytes)
    start_c, done_c = chunk_arrivals(c_size, bw_bytes_per_us, fixed_us, issue_us)
    cu = np.asarray(compute_us, dtype=np.int64)
    l_start, l_end, l_stall = stall_profile(ptr, done_c, cu)
    L = int(ptr.shape[0]) - 1
    C = int(c_size.shape[0])
    if C:
        consume = l_end[c_layer]
        early = (ev >= 0) & (start_c < consume[ev])
        conflict_chunks = np.nonzero(early)[0].astype(np.int64)
    else:
        conflict_chunks = np.zeros(0, dtype=np.int64)
    bw = int(bw_bytes_per_us)
    cost = (c_size + bw - 1) // bw + int(fixed_us) if C else np.zeros(0, np.int64)
    return {"n_chunks": C, "n_layers": L,
            "bytes_total": int(np.asarray(layer_bytes, dtype=np.int64).sum()),
            "total_us": int(l_end[L - 1]) if L else 0,
            "stall_us": int(l_stall.sum()) if L else 0,
            "bubbles": int((l_stall > 0).sum()) if L else 0,
            "conflicts": int(conflict_chunks.shape[0]),
            "link_busy_us": int(cost.sum()) if C else 0,
            "last_arrival_us": int(done_c[C - 1]) if C else 0,
            "window": window, "layer_stall": l_stall, "layer_start": l_start,
            "slot_reuse": reuse, "conflict_chunks": conflict_chunks}

