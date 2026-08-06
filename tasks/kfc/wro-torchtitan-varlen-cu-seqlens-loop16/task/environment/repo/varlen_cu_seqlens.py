"""Variable-length attention metadata for packed-document batches.

Scope module (editable): ``varlen_cu_seqlens.py``.

Given a per-token ``positions`` tensor of shape ``[batch, seq_len]`` whose values
reset to 0 at each packed-document start and increase by 1 within a document, build
the *cumulative sequence length* index ``cu_seqlens`` that a variable-length
(FlashAttention-style) attention kernel consumes, plus ``max_seqlen`` (the longest
document length across the whole batch).

``cu_seqlens`` is expressed over the whole batch flattened row-major: token
``(b, col)`` has global index ``b * seq_len + col``. For every document start
``positions[b][col] == 0`` its global index is emitted (in row-major order), and the
total token count ``batch * seq_len`` is appended as the final entry. The consecutive
differences of ``cu_seqlens`` are exactly the per-document lengths.

The public entry point is::

    build_varlen_cu_seqlens(positions) -> (cu_seqlens: list[int], max_seqlen: int)

``positions`` is a 2-D array-like (list-of-lists / numpy array / anything indexable
as ``positions[b][col]``) of non-negative ints, with ``positions[b][0] == 0`` (every
packed row begins a document).

The current implementation is functionally correct but slow: it scans every token in
Python to find the per-row document starts, assembles the index lists element by
element, and finds ``max_seqlen`` with a Python max loop. Its cost is O(batch *
seq_len) in the interpreter.
"""


def _as_2d_int(positions):
    """Normalize ``positions`` to a list-of-lists of Python ints (rows)."""
    rows = []
    for b in range(len(positions)):
        row = positions[b]
        rows.append([int(v) for v in row])
    return rows


def build_varlen_cu_seqlens(positions):
    """Build packed varlen ``(cu_seqlens, max_seqlen)`` from reset-at-boundary positions.

    Slow-but-correct reference: per-row Python boundary scan + element-wise list
    assembly + Python max.
    """
    pos = _as_2d_int(positions)
    batch_size = len(pos)
    seq_len = len(pos[0]) if batch_size else 0

    cu_seqlens = []
    all_seq_lengths = []
    offset = 0
    for b in range(batch_size):
        row = pos[b]
        # Per-token scan for document starts (positions reset to 0).
        doc_starts = []
        for i in range(seq_len):
            if row[i] == 0:
                doc_starts.append(i)
        # sample_cu_seqlens = doc_starts followed by the row length; the pairwise
        # differences are this row's per-document lengths.
        sample = doc_starts + [seq_len]
        for k in range(len(sample) - 1):
            all_seq_lengths.append(sample[k + 1] - sample[k])
        # Global (batch-flattened) document-start indices for this row.
        for d in doc_starts:
            cu_seqlens.append(d + offset)
        offset += seq_len

    # Final total token count closes the packed cu_seqlens.
    cu_seqlens.append(offset)

    max_seqlen = 0
    for length in all_seq_lengths:
        if length > max_seqlen:
            max_seqlen = length

    return cu_seqlens, int(max_seqlen)
