#!/usr/bin/env python3
"""Standalone verifier workload for the varlen cu_seqlens metadata subsystem
(scope: /app/repo/varlen_cu_seqlens.py :: build_varlen_cu_seqlens).

Drives the scope function STANDALONE on CPU (no torch, no GPU): given a
``[batch, seq_len]`` positions tensor whose values reset to 0 at each packed-document
start, it returns ``(cu_seqlens, max_seqlen)`` — the batch-flattened cumulative
sequence-length index a variable-length attention kernel consumes.

  correctness : compare the scope's ``(cu_seqlens, max_seqlen)`` against an
                INDEPENDENT reference computed here (NOT part of the editable scope)
                directly from the document lengths that generated the positions. Both
                the full cu_seqlens list and max_seqlen must match exactly. Any
                deviation scores 0.
  timing      : warmup + timed repeats on a large batch with many short documents so
                the per-token Python boundary scan + element-wise list assembly + the
                Python max loop dominate and separate from the vectorized
                flatnonzero + diff. The gap GROWS with batch * seq_len and with the
                number of documents. CPU time (process_time) is used so the ratio is
                robust to OS descheduling under fleet load.

Emits one line `WRO_VARLEN_RESULT {json}`.
"""
import gc
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# correctness: small batch, several documents per row, distinct doc lengths.
C_BATCH = 6
C_SEQ = 128
C_SEED = 3
# timing: large batch, long sequence, many short documents (so #boundaries is large
# and the per-token Python scan + per-boundary assembly dominate).
T_BATCH = 96
T_SEQ = 2048
T_DOC_MIN = 8
T_DOC_MAX = 40
T_SEED = 1
WARMUP = 2
ITERS = 5


def load_scope():
    import varlen_cu_seqlens as s
    return s


def _random_doc_lengths(seq_len, dmin, dmax, rng):
    """Partition ``seq_len`` into a list of document lengths in [dmin, dmax]."""
    lengths = []
    remaining = seq_len
    while remaining > 0:
        if remaining <= dmax:
            lengths.append(remaining)
            break
        L = int(rng.integers(dmin, dmax + 1))
        L = min(L, remaining)
        lengths.append(L)
        remaining -= L
    return lengths


def _make_positions(batch, seq_len, dmin, dmax, seed):
    """Build a ``[batch, seq_len]`` positions array (reset-to-0 per document) plus the
    per-row document lengths (ground truth, used by the independent reference)."""
    rng = np.random.default_rng(seed)
    pos = np.empty((batch, seq_len), dtype=np.int64)
    doc_lengths_per_row = []
    for b in range(batch):
        lens = _random_doc_lengths(seq_len, dmin, dmax, rng)
        col = 0
        for L in lens:
            pos[b, col:col + L] = np.arange(L, dtype=np.int64)
            col += L
        doc_lengths_per_row.append(lens)
    return pos, doc_lengths_per_row


def _reference(doc_lengths_per_row, seq_len):
    """Independent reference (deliberately NOT the scope implementation): build the
    batch-flattened cu_seqlens and max_seqlen directly from the document lengths."""
    cu = []
    max_seqlen = 0
    offset = 0
    for lens in doc_lengths_per_row:
        col = 0
        for L in lens:
            cu.append(offset + col)
            col += L
            if L > max_seqlen:
                max_seqlen = L
        offset += seq_len
    cu.append(offset)
    return cu, int(max_seqlen)


def _correctness_case(s):
    pos, doc_lengths = _make_positions(C_BATCH, C_SEQ, 5, 30, C_SEED)
    got_cu, got_max = s.build_varlen_cu_seqlens(pos.tolist())
    ref_cu, ref_max = _reference(doc_lengths, C_SEQ)
    got_cu = [int(x) for x in got_cu]
    cu_match = (got_cu == ref_cu)
    max_match = (int(got_max) == ref_max)
    return {"correctness_ok": bool(cu_match and max_match),
            "cu_match": bool(cu_match), "max_match": bool(max_match),
            "n_boundaries": len(ref_cu) - 1, "batch": C_BATCH, "seq": C_SEQ,
            "module": s.__file__}


def _timing_case(s):
    pos, _ = _make_positions(T_BATCH, T_SEQ, T_DOC_MIN, T_DOC_MAX, T_SEED)
    pos_list = pos.tolist()

    def once():
        s.build_varlen_cu_seqlens(pos_list)

    gc_was = gc.isenabled()
    gc.disable()
    try:
        for _ in range(WARMUP):
            once()
        ts = []
        for _ in range(ITERS):
            t0 = time.process_time()
            once()
            ts.append((time.process_time() - t0) * 1000.0)
    finally:
        if gc_was:
            gc.enable()
    return statistics.median(ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    s = load_scope()
    if mode == "correctness":
        res = _correctness_case(s)
        res["mode"] = "correctness"
        print("WRO_VARLEN_RESULT " + json.dumps(res))
        sys.exit(0 if res["correctness_ok"] else 3)
    elif mode == "timing":
        ms = _timing_case(s)
        print("WRO_VARLEN_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS,
            "batch": T_BATCH, "seq": T_SEQ, "module": s.__file__}))
        sys.exit(0)
    else:
        print("WRO_VARLEN_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
