# Performance Optimization Task — submission entry point.
#
# Implement `select_offload` to the contract in instruction.md, then make it as fast as possible.
# This is the ONLY file you edit. Leaving the NotImplementedError in place scores 0.
import numpy as np  # noqa: F401  (available; you may use it)


def select_offload(sketch, seeds, keys, present, threshold):
    """Select which KV blocks a tiered cache should offload, using a count-min frequency sketch.

    A hierarchical KV-cache offload policy (TinyLFU-style) decides whether each candidate block is
    "hot" enough to promote to the next tier. Block frequencies are tracked in a count-min sketch:
    a `[D, W]` counter table with `D` independent hash rows. A block's estimated frequency is the
    **minimum** counter across its `D` hashed columns (the count-min estimate — the minimum is the
    tightest upper bound, since collisions only ever inflate a row). A block is admitted only if it
    is frequent enough AND not already present in the destination tier.

    Contract (deterministic; all correct implementations agree exactly):
      * `sketch`: a 2-D `numpy` int64 array of shape `[D, W]` — the count-min counters.
      * `seeds`: a 1-D `numpy` int64 array of length `D` — the per-row hash multipliers.
      * `keys`: a 1-D `numpy` int64 array of length `N` — the candidate block hash keys
        (each in `[0, 2**31)`).
      * `present`: a 1-D `numpy` int8/int array of length `N` — `present[i] == 1` means block `i`
        is already in the destination tier (must NOT be offloaded); `0` means absent.
      * `threshold`: an int — the LFU count threshold.
      * For block `i`, its column in row `d` is `col = (keys[i] * seeds[d]) % W`, and its estimated
        frequency is `est[i] = min over d of sketch[d, col]`.
      * Block `i` is **admitted** iff `est[i] > threshold` AND `present[i] == 0`.
      * Return the indices of all admitted blocks, in **ascending index order**, as a 1-D `numpy`
        int64 array.

    Args:
        sketch:    int64 numpy array [D, W] count-min counters.
        seeds:     int64 numpy array [D] per-row hash multipliers.
        keys:      int64 numpy array [N] candidate block hash keys in [0, 2**31).
        present:   int numpy array [N] destination-presence flags (1 = present).
        threshold: int LFU count threshold.

    Return:
        numpy.ndarray[int64]: ascending indices of admitted blocks.
    """
    raise NotImplementedError("implement select_offload to the contract in instruction.md")


def custom_kernel(data):
    """Entry point the verifier calls. data = (sketch, seeds, keys, present, threshold). Already
    wired — returns the ascending admitted-block index array from select_offload."""
    sketch, seeds, keys, present, threshold = data
    return select_offload(sketch, seeds, keys, present, threshold)
