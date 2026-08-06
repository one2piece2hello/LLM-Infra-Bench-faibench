# Performance Optimization Task — submission entry point.
#
# Implement `merge_shard_extents` to the contract in instruction.md, then make it as fast as
# possible. This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import numpy as np  # noqa: F401  (available; you may use it)


def merge_shard_extents(entries, num_tensors):
    """Merge distributed-checkpoint shard metadata across ranks into per-tensor global extents.

    In a distributed checkpoint, every rank independently writes shard-metadata entries describing
    the byte range it owns for each logical tensor. To assemble the global metadata you must, for
    each logical tensor, compute the total flattened size = the largest end offset any shard
    reaches for that tensor.

    Contract (deterministic; all correct implementations agree exactly):
      * ``entries``: a 2-D ``numpy`` int64 array of shape ``(N, 3)``; each row is one shard record
        ``[tensor_id, offset, size]`` with ``0 <= tensor_id < num_tensors``, ``offset >= 0``,
        ``size >= 1``. The rows are in ARBITRARY order (ranks interleave).
      * Each shard's END offset is ``offset + size``.
      * ``num_tensors`` (int ``G``): every tensor id in ``[0, G)`` appears in at least one row.
      * For each tensor id ``t`` in ``[0, G)``, its global extent is the MAXIMUM ``offset + size``
        over all rows with ``tensor_id == t``.
      * Return a 1-D ``numpy`` int64 array of length ``G``: the global extent per tensor id,
        indexed by tensor id (``out[t]`` = extent of tensor ``t``).

    Args:
        entries:     (N, 3) int64 numpy array of [tensor_id, offset, size] shard records.
        num_tensors: number of logical tensors G (ids 0..G-1, each present at least once).

    Return:
        numpy.ndarray[int64] of shape (G,): global extent (max end offset) per tensor id.
    """
    n = entries.shape[0]
    buf = _scratch
    out = np.zeros(num_tensors, dtype=np.int64)
    # Cache-friendly chunked scatter-max: compute end offsets into a small reusable
    # buffer, then scatter the per-tensor max while the chunk is still hot in cache.
    add = _ADD
    maxat = _MAXAT
    for lo in range(0, n, _CHUNK):
        hi = lo + _CHUNK
        if hi > n:
            hi = n
        blk = entries[lo:hi]
        b = buf[: hi - lo]
        add(blk[:, 1], blk[:, 2], out=b)
        maxat(out, blk[:, 0], b)
    return out


_CHUNK = 24576
_scratch = np.empty(_CHUNK, dtype=np.int64)
_ADD = np.add
_MAXAT = np.maximum.at


def custom_kernel(data):
    """Entry point the verifier calls. data = (entries, num_tensors). Already wired — returns the
    per-tensor global-extent array from merge_shard_extents."""
    entries, num_tensors = data
    return merge_shard_extents(entries, num_tensors)
