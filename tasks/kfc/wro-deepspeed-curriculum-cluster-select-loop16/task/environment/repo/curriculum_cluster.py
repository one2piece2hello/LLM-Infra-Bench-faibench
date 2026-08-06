"""Curriculum-learning difficulty-cluster selection for large-scale training stability.

DeepSpeed's data-efficiency curriculum learning stabilises very-large-scale
pretraining by feeding samples in an easy-to-hard order: samples are pre-bucketed
by a difficulty *metric* into per-difficulty-value rows (``index_to_sample[v]`` =
the array of sample-ids whose metric value is ``index_to_metric[v]``), sorted by
increasing difficulty. At each curriculum step the trainer selects the flat set of
sample-ids that fall inside a difficulty window, either

  * VALUE-based:      every row whose metric value is in ``(value_start, value_end]``, or
  * PERCENTILE-based: the contiguous slice of samples covering the fraction of the
                      one-epoch population in ``[pct_start, pct_end)`` of ``num_bins``
                      equal-count percentile bands (with a running-count walk that
                      slices the two boundary rows partially and takes whole interior
                      rows, stopping once the end count is reached).

``select_curriculum_cluster`` returns the concatenated 1-D int array of selected
sample-ids (in row-then-within-row order). This is the functionally-correct but slow
reference: it walks the rows in a Python loop and grows the result with a fresh
``numpy.concatenate`` per matching row (each append re-copies the whole running
buffer, so the row walk is quadratic in the number of selected rows) and re-sums the
one-epoch population from scratch. The public signature and the returned array are
the behavioural contract.

Mirrors DeepSpeed ``deepspeed/runtime/data_pipeline/data_sampling/data_sampler.py``
``DeepSpeedDataSampler.get_sample_based_on_metric_value`` /
``get_sample_based_on_metric_percentile`` (Apache-2.0).
"""

import numpy as np


def select_curriculum_cluster(
    index_to_sample,
    index_to_metric,
    mode,
    lo,
    hi,
    num_bins=None,
):
    """Select the flat cluster of sample-ids for one curriculum difficulty window.

    :param index_to_sample: list of 1-D numpy int arrays; ``index_to_sample[r]`` holds
        the sample-ids bucketed at difficulty-row ``r`` (rows sorted easy->hard).
    :param index_to_metric: 1-D numpy array; ``index_to_metric[r]`` is the difficulty
        metric value of row ``r`` (non-decreasing in ``r``).
    :param mode: ``"value"`` or ``"percentile"``.
    :param lo: window lower bound. VALUE mode: exclusive metric lower bound
        (``value_start``). PERCENTILE mode: inclusive percentile-band start
        (``pct_start``, an int in ``[0, num_bins]``).
    :param hi: window upper bound. VALUE mode: inclusive metric upper bound
        (``value_end``). PERCENTILE mode: exclusive percentile-band end
        (``pct_end``, an int in ``[0, num_bins]``); when ``hi == num_bins`` the band
        extends to the exact end of the population.
    :param num_bins: number of equal-count percentile bands (PERCENTILE mode only).
    :returns: 1-D numpy int64 array of the selected sample-ids (row-then-within-row
        order), or an empty int64 array when nothing is selected.
    """
    metric = np.asarray(index_to_metric)
    n_rows = len(index_to_sample)

    if mode == "value":
        # NAIVE_VALUE_ROWWALK: Python loop, re-copy+concat the running buffer per hit.
        new_samples = None
        for row in range(n_rows):
            if metric[row] <= hi and metric[row] > lo:
                row_samples = np.copy(np.asarray(index_to_sample[row]))
                new_samples = (
                    row_samples
                    if new_samples is None
                    else np.concatenate((new_samples, row_samples), axis=None)
                )
        if new_samples is None:
            return np.array([], dtype=np.int64)
        return new_samples.astype(np.int64, copy=False)

    if mode == "percentile":
        # NAIVE_PERCENTILE_ROWWALK: re-sum the epoch population, then a running-count
        # walk that partially slices the two boundary rows and re-copies+concats each.
        one_epoch_size = 0
        for row in range(n_rows):
            one_epoch_size += len(index_to_sample[row])
        sample_per_pct = one_epoch_size // num_bins
        start_count = sample_per_pct * lo
        end_count = sample_per_pct * hi
        if hi == num_bins:
            end_count = one_epoch_size

        new_samples = None
        current_count = 0
        for row in range(n_rows):
            row_size = len(index_to_sample[row])
            if current_count + row_size > start_count:
                row_start = max(0, start_count - current_count)
                if current_count + row_size <= end_count:
                    row_end = row_size
                else:
                    row_end = end_count - current_count
                row_samples = np.copy(np.asarray(index_to_sample[row])[row_start:row_end])
                new_samples = (
                    row_samples
                    if new_samples is None
                    else np.concatenate((new_samples, row_samples), axis=None)
                )
            current_count += row_size
            if current_count >= end_count:
                break
        if new_samples is None:
            return np.array([], dtype=np.int64)
        return new_samples.astype(np.int64, copy=False)

    raise ValueError("mode must be 'value' or 'percentile'")
