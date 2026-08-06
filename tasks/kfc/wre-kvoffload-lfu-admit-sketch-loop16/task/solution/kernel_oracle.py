# ORACLE variant (reviewer-only; never baked into the image). FAST O(N*D) fully-vectorized implementation:
# broadcast the D per-row hashes over all N keys at once, gather every block's D counters with a
# single take_along_axis, reduce with a vectorized min over the row axis (the count-min estimate),
# then select admitted blocks with a boolean mask. No python per-block loop.
import numpy as np


def select_offload(sketch, seeds, keys, present, threshold):
    sketch = np.asarray(sketch)
    seeds = np.asarray(seeds).astype(np.int64)
    keys = np.asarray(keys).astype(np.int64)
    present = np.asarray(present)
    W = int(sketch.shape[1])
    cols = (keys[None, :] * seeds[:, None]) % W        # [D, N] hashed columns (values bounded, no overflow)
    gathered = np.take_along_axis(sketch, cols, axis=1)  # [D, N] counters
    est = gathered.min(axis=0)                           # [N] count-min estimate = min across rows
    admit = (est > threshold) & (present == 0)
    return np.nonzero(admit)[0].astype(np.int64)


def custom_kernel(data):
    sketch, seeds, keys, present, threshold = data
    return select_offload(sketch, seeds, keys, present, threshold)
