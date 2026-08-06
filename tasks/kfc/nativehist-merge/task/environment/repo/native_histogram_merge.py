"""Merge of many sparse exponential-bucket histograms into one.

Public entry points:
    ``NativeHistogramMerger(schema)`` — an accumulator that folds histograms together.
    ``NativeHistogramMerger.add(histogram)`` — fold one sparse histogram into the merge.
    ``NativeHistogramMerger.merged()`` -> the single merged histogram (a dict).

Domain
------
An exponential-bucket histogram summarizes a distribution of observations over a
fixed *log-scale* bucket schema: bucket ``i`` covers the value range
``(base**i, base**(i+1)]`` for a base fixed by the schema (a finer schema packs
more, narrower buckets per decade). Because real observations cluster, most
buckets are empty, so a histogram is stored **sparsely** as a sorted list of
``(bucket_index, count)`` pairs — only the populated buckets — plus a separate
``zero_count`` (observations at/near zero, which have no finite log bucket) and a
``sum`` aggregate. A metrics collector or query engine folds ("merges") many such
per-series histograms into one aggregate; this file implements that merge.

Data model
----------
A histogram is a ``dict`` with:
  * ``"schema"``: an ``int`` resolution tag. All histograms in one merge must
    share the same schema (see the error contract).
  * ``"zero_count"``: a non-negative ``int`` — observations in the zero bucket.
  * ``"sum"``: an ``int`` aggregate that is merged additively.
  * ``"buckets"``: a list of ``(bucket_index, count)`` pairs, **sorted ascending by
    bucket_index**, with distinct indices and each ``count`` a positive ``int``.
    ``bucket_index`` is a signed ``int`` (may be far negative to far positive).
Missing ``zero_count`` / ``sum`` / ``buckets`` default to ``0`` / ``0`` / ``[]``.

Contract
--------
``NativeHistogramMerger(schema)``: ``schema`` must be an ``int`` (``bool`` is
rejected). ``merged()`` on a merger with nothing added returns the empty histogram
``{"schema": schema, "zero_count": 0, "sum": 0, "buckets": []}``.

``add(histogram)``: fold ``histogram`` into the running merge. Its ``"schema"``
must equal the merger's schema, otherwise ``ValueError`` (a coarser/finer histogram
is a different bucket geometry and cannot be summed bucket-for-bucket here).

``merged()`` -> a histogram dict whose:
  * ``"schema"`` is the merger's schema;
  * ``"zero_count"`` is the sum of every added histogram's ``zero_count``;
  * ``"sum"`` is the sum of every added histogram's ``sum``;
  * ``"buckets"`` is the per-index count total over the **union** of all populated
    bucket indices — for each index present in any added histogram, the sum of its
    counts — emitted sorted ascending by index, dropping any index whose total is 0.

The merge is associative and commutative: the order in which histograms are added,
and any grouping of the adds, never changes the result. Total observation count is
conserved: ``merged.zero_count + sum(count for _, count in merged.buckets)`` equals
the same total summed over all inputs.

Why the current implementation is slow
--------------------------------------
This accumulates into a **dense array spanning the entire index range** min..max
of every populated bucket, then re-sparsifies by walking that whole range. When the
populated buckets are few but spread far apart (the common case — a wide dynamic
range with observations clustered in a handful of buckets), the dense array is
mostly zeros and the re-sparsify walk visits a huge number of empty cells. The cost
grows with the index *range*, not with the number of populated buckets. Make the
merge faster (work that scales with the populated buckets, not the index span)
while keeping the contract above exact.

Note on allowed operations
--------------------------
Implement the sparse merge yourself. Do not delegate the merge to an exponential /
native-histogram library or a sketch package, or to a vectorized numeric array
library — the scoring harness scans the submitted file for those and scores the
task 0 (do not reference them even in comments). Build the bucket alignment and the
count summation yourself from standard-library primitives.
"""


def _check_schema(schema):
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise TypeError(f"schema must be an int, got {type(schema).__name__}")


def _normalize(histogram):
    """Read a histogram dict into (schema, zero_count, sum, buckets_list)."""
    schema = histogram["schema"]
    zero_count = int(histogram.get("zero_count", 0))
    total = int(histogram.get("sum", 0))
    buckets = [(int(i), int(c)) for i, c in histogram.get("buckets", [])]
    return schema, zero_count, total, buckets


class NativeHistogramMerger:
    """See the module docstring for the full contract.

    Naive reference accumulator: stores every added histogram, and on ``merged()``
    expands them into one dense array covering the full min..max index range, adds
    every count into it, then re-sparsifies by walking the entire range. Correct,
    but the work grows with the index span rather than with the populated buckets.
    """

    def __init__(self, schema=0):
        _check_schema(schema)
        self.schema = int(schema)
        # each stored item: (zero_count, sum, [(index, count), ...])
        self._hists = []

    def add(self, histogram):
        schema, zero_count, total, buckets = _normalize(histogram)
        if schema != self.schema:
            raise ValueError(
                f"schema mismatch: cannot fold schema {schema} into a merge at "
                f"schema {self.schema}")
        self._hists.append((zero_count, total, buckets))

    def merged(self):
        total_zero = 0
        total_sum = 0
        lo = None
        hi = None
        for zero_count, total, buckets in self._hists:
            total_zero += zero_count
            total_sum += total
            for idx, _cnt in buckets:
                if lo is None or idx < lo:
                    lo = idx
                if hi is None or idx > hi:
                    hi = idx

        out_buckets = []
        if lo is not None:
            width = hi - lo + 1
            # dense array over the ENTIRE index range (mostly zeros when sparse)
            dense = [0] * width
            for _zero_count, _total, buckets in self._hists:
                for idx, cnt in buckets:
                    dense[idx - lo] += cnt
            # re-sparsify: walk the whole dense range, emit the non-empty cells
            for offset in range(width):
                c = dense[offset]
                if c != 0:
                    out_buckets.append((lo + offset, c))

        return {
            "schema": self.schema,
            "zero_count": total_zero,
            "sum": total_sum,
            "buckets": out_buckets,
        }
