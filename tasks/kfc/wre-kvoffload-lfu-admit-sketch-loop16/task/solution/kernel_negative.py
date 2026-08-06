# NEGATIVE variant (reviewer-only; never baked into the image). FAST-but-WRONG.
# Breaks the count-min estimate invariant: it reduces the D hashed counters with MAX instead of
# MIN. The count-min estimate MUST be the minimum across rows (collisions only inflate counters, so
# the min is the tightest upper bound). Taking the max over-estimates every block's frequency, so
# blocks whose true (min) estimate is <= threshold but whose max row counter exceeds it are wrongly
# admitted -> the admitted set differs from the contract and correctness FAILS.
import numpy as np


def select_offload(sketch, seeds, keys, present, threshold):
    sketch = np.asarray(sketch)
    seeds = np.asarray(seeds).astype(np.int64)
    keys = np.asarray(keys).astype(np.int64)
    present = np.asarray(present)
    W = int(sketch.shape[1])
    cols = (keys[None, :] * seeds[:, None]) % W
    gathered = np.take_along_axis(sketch, cols, axis=1)
    est = gathered.max(axis=0)                           # BUG: max across rows, not min
    admit = (est > threshold) & (present == 0)
    return np.nonzero(admit)[0].astype(np.int64)


def custom_kernel(data):
    sketch, seeds, keys, present, threshold = data
    return select_offload(sketch, seeds, keys, present, threshold)
