"""KV-cache commit for the accepted prefix of a verified draft tree (CPU, numpy only).

After a target model has verified a batch of speculative draft trees, the accepted
tokens of every request have to be *committed*: their key/value rows, which the draft
pass wrote into scratch slots, must be moved into the slots the request's own KV page
table points at, the per-request bookkeeping (sequence length, the next-step filter,
the bonus token that seeds the following draft round) has to be brought forward, and
all of it has to survive the fact that the accepted run length differs from request to
request.

The batch is described by four ragged pieces:

* a page table ``req_to_token`` plus the per-request row ids ``req_pool_indices``,
* the current ``seq_lens`` and the per-request count of correct drafts,
* an ``accept_index`` table of scratch-slot positions with ``-1`` marking "not
  accepted", and the scratch slot list ``out_cache_loc`` those positions index into.

Two entry points are exposed:

* ``plan_accept_move`` -- build the move plan and the sequence bookkeeping only.
* ``commit_verified_step`` -- the same plan plus the bonus tokens, the next-step
  filter and the actual KV row movement.

Both are thin wrappers over the single core ``_accept_core``.
"""

import numpy as np

__all__ = ["plan_accept_move", "commit_verified_step", "MAX_BS", "MAX_WIDTH"]

MAX_BS = 4096                       # batch cap
MAX_WIDTH = 256                     # accept-table width cap (draft tokens per request)


def plan_accept_move(req_pool_indices, req_to_token, seq_lens, num_correct_drafts,
                     accept_index, out_cache_loc):
    """Build the KV move plan for one verified step.

    Parameters
    ----------
    req_pool_indices : ndarray, shape (bs,), dtype int64
        For each request, its row in ``req_to_token``.
    req_to_token : ndarray, shape (n_rows, pool_len), dtype int64
        The KV page table: ``req_to_token[r, t]`` is the pool slot holding position
        ``t`` of the request living in row ``r``.
    seq_lens : ndarray, shape (bs,), dtype int64
        Per-request committed length before this step.
    num_correct_drafts : ndarray, shape (bs,), dtype int64
        Per-request number of accepted draft tokens, EXCLUDING the bonus token.
    accept_index : ndarray, shape (bs, width), dtype int64
        Positions into ``out_cache_loc`` of the accepted tokens, ``-1`` where there is
        none.
    out_cache_loc : ndarray, shape (n_scratch,), dtype int64
        The scratch pool slots the verification pass wrote, in draft-tree order.

    Returns
    -------
    dict
        The five plan keys documented on :func:`_accept_core`
        (``tgt_cache_loc``, ``accept_out_cache_loc``, ``n_move``, ``n_accept``,
        ``seq_lens_next``) plus ``num_accept_tokens_filter``, ``bonus_tokens`` and
        ``kv_cache``, all three ``None`` in this flavour.

    Raises
    ------
    ValueError
        On any contract violation listed in :func:`_accept_core`.
    """
    return _accept_core(req_pool_indices, req_to_token, seq_lens, num_correct_drafts,
                        accept_index, out_cache_loc, None, None, None)


def commit_verified_step(req_pool_indices, req_to_token, seq_lens, num_correct_drafts,
                         accept_index, out_cache_loc, accept_tokens, unfinished_index,
                         kv_cache):
    """Plan the move, then apply it and produce the per-request bookkeeping.

    Parameters
    ----------
    req_pool_indices, req_to_token, seq_lens, num_correct_drafts, accept_index,
    out_cache_loc
        As on :func:`plan_accept_move`.
    accept_tokens : ndarray, shape (bs, width), dtype int64, or None
        The accepted token ids laid out like ``accept_index``.  ``None`` disables the
        ``bonus_tokens`` output.
    unfinished_index : ndarray, shape (n_unfinished,), dtype int64, or None
        The requests that continue into the next draft round.  ``None`` disables the
        ``num_accept_tokens_filter`` output.  May be empty.
    kv_cache : ndarray, shape (n_pool, dim), dtype float32, or None
        The KV pool to move rows inside.  ``None`` disables the movement; the plan is
        still produced.

    Returns
    -------
    dict
        All eight keys documented on :func:`_accept_core`.

    Raises
    ------
    ValueError
        On any contract violation listed in :func:`_accept_core`.
    """
    return _accept_core(req_pool_indices, req_to_token, seq_lens, num_correct_drafts,
                        accept_index, out_cache_loc, accept_tokens, unfinished_index,
                        kv_cache)


def _accept_core(req_pool_indices, req_to_token, seq_lens, num_correct_drafts,
                 accept_index, out_cache_loc, accept_tokens, unfinished_index,
                 kv_cache):
    """The single commit core shared by both public entry points.

    Notation
    --------
    ``bs`` is ``accept_index.shape[0]``, ``width`` is ``accept_index.shape[1]``,
    ``pool_len`` is ``req_to_token.shape[1]``, and

        accept_lens[j] = num_correct_drafts[j] + 1        # the +1 is the bonus token

    Validation (all of it before any output is produced; ``ValueError`` on failure)
    ------------------------------------------------------------------------------
    * ``accept_index`` is a 2-D int64 array with ``1 <= bs <= MAX_BS`` and
      ``1 <= width <= MAX_WIDTH``;
    * ``req_to_token`` is a 2-D int64 array with at least one row and one column;
    * ``req_pool_indices``, ``seq_lens`` and ``num_correct_drafts`` are 1-D int64
      arrays of shape ``(bs,)``;
    * ``out_cache_loc`` is a 1-D int64 array with at least one entry;
    * every ``req_pool_indices`` entry is in ``[0, req_to_token.shape[0])``;
    * every ``seq_lens`` entry and every ``num_correct_drafts`` entry is ``>= 0``;
    * ``num_correct_drafts[j] + 1 <= width`` for every request (the accepted run,
      bonus included, has to fit in the accept table);
    * ``seq_lens[j] + num_correct_drafts[j] + 1 <= pool_len`` for every request (the
      destination slice has to fit in the page-table row);
    * every ``accept_index`` entry is either ``-1`` or in ``[0, out_cache_loc.size)``;
    * ``accept_tokens``, when not ``None``, is an int64 array of exactly
      ``accept_index``'s shape;
    * ``unfinished_index``, when not ``None``, is a 1-D int64 array (possibly empty)
      whose entries are all in ``[0, bs)``;
    * ``kv_cache``, when not ``None``, is a 2-D float32 array with at least one row
      and one column, and both every ``out_cache_loc`` entry and every planned
      destination (every ``tgt_cache_loc[:n_move]`` entry, see below) must be in
      ``[0, kv_cache.shape[0])``.

    Behaviour
    ---------
    1. **The destination plan.**  ``offsets`` is the EXCLUSIVE prefix sum of
       ``accept_lens``, ``n_move = int(accept_lens.sum())``, and ``tgt_cache_loc`` is a
       zero-filled ``(bs * width,)`` int64 buffer into which every request writes its
       page-table slice::

           tgt_cache_loc[offsets[j] : offsets[j] + accept_lens[j]]
               = req_to_token[req_pool_indices[j],
                              seq_lens[j] : seq_lens[j] + accept_lens[j]]

       Entries at and beyond ``n_move`` stay zero.  Note that the buffer is sized by
       ``bs * width`` but only ``n_move <= bs * width`` entries are ever written, so
       the plan is dense at the front.

    2. **The source compaction.**  Read ``accept_index`` flat in row-major order.  A
       flat position ``q`` survives when ``accept_index.reshape(-1)[q] != -1``, and it
       lands at ``dst[q]`` = the number of surviving positions STRICTLY BEFORE ``q``
       -- a single global running count over the whole flattened table, not a
       per-request one::

           accept_out_cache_loc[dst[q]] = out_cache_loc[flat[q]]   for surviving q

       ``accept_out_cache_loc`` is likewise a zero-filled ``(bs * width,)`` int64
       buffer, and ``n_accept`` is the total number of survivors.  Sentinels may sit
       anywhere in a row and a request's survivor count need not equal its
       ``accept_lens``; the definition above is total either way.

    3. **Sequence bookkeeping.**  ``seq_lens_next = seq_lens + accept_lens``, returned
       as a new array -- ``seq_lens`` itself is NOT modified, and neither is any other
       input.

    4. **The next-step filter.**  When ``unfinished_index`` is ``None`` the output
       ``num_accept_tokens_filter`` is ``None``.  Otherwise it is a zero-filled
       ``(bs,)`` int64 array with ``filter[unfinished_index] = accept_lens[
       unfinished_index]``; requests absent from ``unfinished_index`` keep their zero.

    5. **The bonus token.**  When ``accept_tokens`` is ``None`` the output
       ``bonus_tokens`` is ``None``.  Otherwise it is an ``(bs,)`` int64 array with
       ``bonus_tokens[j] = accept_tokens[j, accept_lens[j] - 1]`` -- the LAST entry of
       the accepted run, which is why ``accept_lens`` and not ``num_correct_drafts``
       indexes it.

    6. **The move.**  When ``kv_cache`` is ``None`` the output ``kv_cache`` is
       ``None``.  Otherwise the returned array is a fresh copy of the incoming pool in
       which, for ``p`` ascending over ``range(n_move)``::

           out[tgt_cache_loc[p]] = incoming[accept_out_cache_loc[p]]

       Exactly ``n_move`` rows are moved -- the plan's length, independent of
       ``n_accept``.  Sources are always read from the INCOMING pool, never from
       partially written output, and if two plan entries share a destination the later
       one wins.

    Returns
    -------
    dict with

    ``tgt_cache_loc``            int64   (bs * width,)
    ``accept_out_cache_loc``     int64   (bs * width,)
    ``n_move``                   int     python int
    ``n_accept``                 int     python int
    ``seq_lens_next``            int64   (bs,)
    ``num_accept_tokens_filter`` int64   (bs,)   or ``None``
    ``bonus_tokens``             int64   (bs,)   or ``None``
    ``kv_cache``                 float32 (n_pool, dim) or ``None``

    No input array is modified.
    """
    raise NotImplementedError("_accept_core is not implemented yet")
