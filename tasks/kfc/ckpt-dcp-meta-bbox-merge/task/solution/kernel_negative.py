# NEGATIVE variant (reviewer-only; not baked into the image). FAST-but-WRONG.
# Breaks the extent invariant: it scatter-reduces the MAX of the raw OFFSET only, forgetting to
# add each shard's size. The end offset of a tensor is offset+size, so dropping size understates
# every extent (and picks the shard with the largest start, not the largest end) -> the returned
# per-tensor extents are wrong -> correctness FAILS.
import numpy as np


def merge_shard_extents(entries, num_tensors):
    e = np.asarray(entries, dtype=np.int64)
    G = int(num_tensors)
    out = np.zeros(G, dtype=np.int64)
    if e.shape[0] == 0:
        return out
    tid = e[:, 0]
    off = e[:, 1]                          # BUG: uses offset alone, drops + size
    np.maximum.at(out, tid, off)
    return out


def custom_kernel(data):
    entries, num_tensors = data
    return merge_shard_extents(entries, num_tensors)
