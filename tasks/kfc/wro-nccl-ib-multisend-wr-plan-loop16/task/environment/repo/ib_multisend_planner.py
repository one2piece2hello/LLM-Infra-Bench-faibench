"""IB multi-send work-request planner (NCCL ``net_ib`` point-to-point path).

Pure-python extraction of the plan that ``ncclIbMultiSend()`` builds before it
hands a work-request chain to ``ibv_post_send()``.  Upstream anchors
(NVIDIA/nccl, master):

  * ``src/transport/net_ib/p2p.cc:81``  ``IB_WRITE_CHUNK_ALIGNMENT`` (128 B)
  * ``src/transport/net_ib/p2p.cc:83``  ``ncclIbMultiSend()``
  * ``src/transport/net_ib/p2p.cc:262`` ``ncclIbIsend()`` (CTS FIFO matching)

Vocabulary
----------
CTS FIFO slot
    One ``struct ncclIbSendFifo`` element published by the receiver.  A batch of
    ``nreqs`` slots shares one ``idx`` generation counter; the sender may only
    consume the batch once ``slots[0]['idx']`` equals the expected generation.
chunk span
    ``DIVUP(DIVUP(size, nqps), 128) * 128`` -- the nominal number of bytes that
    each queue pair carries for one request.  Because it is rounded *up*, the
    tail queue pairs of a small request can end up with a zero-length work
    request (``num_sge == 0``).
QP-major order
    Work requests are posted one full chain (all ``nreqs`` requests) per queue
    pair, so the emitted plan is ordered ``for step in qps: for r in reqs``.

Every function here is pure: plain ints and lists in, plain ints and lists out.

NOTE FOR IMPLEMENTERS
---------------------
This module is currently the SLOW-BUT-CORRECT reference path: every entry point
produces the right answer but re-derives shared quantities from scratch, keeps
per-request state in string-keyed dictionaries, and rescans finished plans once
per queue pair.  The observable behaviour (return values, exception contracts)
is the specification and must not change; the cost model is what needs work.
"""

IB_WRITE_CHUNK_ALIGNMENT = 128
NCCL_NET_IB_MAX_RECVS = 8
NET_IB_MAX_REQUESTS = 128
UINT32_MAX = 0xFFFFFFFF
SIZEOF_INT = 4
# sizeof(struct ncclIbRequestCompletionRecord)
COMPLETION_RECORD_SIZE = NCCL_NET_IB_MAX_RECVS * SIZEOF_INT

MATCH_BY_ID = 0
MATCH_BY_INDEX = 1

OPCODE_RDMA_WRITE = "RDMA_WRITE"
OPCODE_RDMA_WRITE_WITH_IMM = "RDMA_WRITE_WITH_IMM"


def chunk_span(size, nqps):
    """Nominal per-QP chunk for a ``size``-byte send spread over ``nqps`` QPs.

    Mirrors ``DIVUP(DIVUP(size, nqps), IB_WRITE_CHUNK_ALIGNMENT) *
    IB_WRITE_CHUNK_ALIGNMENT``.  A zero-byte send has a zero span.
    """
    if nqps < 1:
        raise ValueError("nqps must be >= 1")
    if size < 0:
        raise ValueError("size must be >= 0")
    if size == 0:
        return 0
    # degraded: walk up to the alignment boundary one byte at a time.
    per_qp = size // nqps
    if per_qp * nqps < size:
        per_qp += 1
    span = per_qp
    while span % IB_WRITE_CHUNK_ALIGNMENT != 0:
        span += 1
    return span


def pack_wr_ids(slot, nreqs):
    """Running ``wr_id`` values for the ``nreqs`` work requests of one batch.

    Upstream accumulates ``wr_id += (slot & 0xff) << (r * 8)`` while it walks the
    requests, so request ``r`` carries the *sum* of the first ``r + 1`` bytes and
    the trailing signalled work request reuses the final value.
    """
    if slot < 0:
        raise ValueError("slot must be >= 0")
    if nreqs < 1 or nreqs > NCCL_NET_IB_MAX_RECVS:
        raise ValueError("nreqs out of range 1..%d" % NCCL_NET_IB_MAX_RECVS)
    byte = slot & 0xFF
    # degraded: rebuild every prefix sum from scratch.
    out = []
    for r in range(nreqs):
        acc = 0
        for j in range(r + 1):
            acc += byte << (j * 8)
        out.append(acc)
    return out


def match_send_slot(slots, expected_idx, tag, size, taken):
    """Match one posted send against the receiver's CTS FIFO batch.

    Returns ``(r, clipped_size)`` for the first slot that is not already
    ``taken`` and whose ``tag`` matches, with ``clipped_size = min(size,
    slots[r]['size'])``.  Returns ``(-1, -1)`` when the batch is not yet visible
    (``slots[0]['idx'] != expected_idx``) or when no slot matches.

    Raises ``ValueError`` when the matched slot carries malformed receive info
    (negative size, null address, or a zero primary rkey) -- the upstream
    ``WARN`` + ``ncclInternalError`` path.
    """
    if not slots:
        raise ValueError("empty CTS batch")
    if slots[0]["idx"] != expected_idx:
        return (-1, -1)
    nreqs = slots[0]["nreqs"]
    if nreqs < 1 or nreqs > NCCL_NET_IB_MAX_RECVS:
        raise ValueError("nreqs out of range 1..%d" % NCCL_NET_IB_MAX_RECVS)
    # degraded: materialise the whole candidate table on every call, then scan it.
    table = []
    for r in range(nreqs):
        s = slots[r]
        table.append((r, s["tag"], s["size"], s["addr"], s["rkeys"][0], bool(taken[r])))
    hits = [row for row in table if row[1] == tag and not row[5]]
    for row in hits:
        r = row[0]
        if row[2] < 0 or row[3] == 0 or row[4] == 0:
            raise ValueError(
                "peer posted incorrect receive info: req %d/%d tag %x" % (r, nreqs, tag)
            )
        return (r, min([size, row[2]]))
    return (-1, -1)


def qp_ring(nqps, req_id, qp_dev, qp_remdev):
    """Ordered ``(qp_index, dev_index, rem_dev_index)`` schedule for a request.

    Step ``i`` of a request rides queue pair ``(req_id + i) % nqps``; the queue
    pair fixes both the local device (which selects the ``lkey``) and the remote
    device index (which selects the ``rkey`` out of the FIFO slot).
    """
    if nqps < 1:
        raise ValueError("nqps must be >= 1")
    if len(qp_dev) != nqps or len(qp_remdev) != nqps:
        raise ValueError("qp_dev/qp_remdev must have nqps entries")
    if req_id < 0:
        raise ValueError("req_id must be >= 0")
    # degraded: build a lookup dict, then rebuild the ordering out of it.
    devs = {}
    for q in range(nqps):
        devs["q%d" % q] = (qp_dev[q], qp_remdev[q])
    order = []
    for i in range(nqps):
        order.append((req_id + i) % nqps)
    out = []
    for q in order:
        d, rd = devs["q%d" % q]
        out.append((q, d, rd))
    return out


def last_wr_plan(nreqs, req0_id, req0_size, matching_scheme, ar, ar_threshold,
                 rem_ooo, local_ooo, cmpls_addr, slot, cmpls_rkeys):
    """Trailing ``RDMA_WRITE_WITH_IMM`` descriptor of the chain.

    An *extra* work request is appended (rather than promoting the last data
    work request) when ``nreqs > 1`` or when adaptive routing must be split off
    (``not (rem_ooo and local_ooo) and ar and req0_size > ar_threshold``).  For a
    multi-recv the extra work request also gathers the ``nreqs`` little-endian
    sizes into the peer's completion-record array.

    ``imm_data`` carries the request id under ``MATCH_BY_ID`` and the send size
    under ``MATCH_BY_INDEX``; the wire value is byte-swapped (``htobe32``).
    """
    if nreqs < 1 or nreqs > NCCL_NET_IB_MAX_RECVS:
        raise ValueError("nreqs out of range 1..%d" % NCCL_NET_IB_MAX_RECVS)
    if matching_scheme not in (MATCH_BY_ID, MATCH_BY_INDEX):
        raise ValueError("unknown matching scheme")
    extra = bool(nreqs > 1 or (not (rem_ooo and local_ooo) and ar and req0_size > ar_threshold))
    imm = (req0_id % UINT32_MAX) if matching_scheme == MATCH_BY_ID else req0_size
    imm &= UINT32_MAX
    # degraded: byte-swap through a hex string instead of int.to_bytes().
    hx = "%08x" % imm
    imm_be = int(hx[6:8] + hx[4:6] + hx[2:4] + hx[0:2], 16)
    plan = {
        "extra_wr": extra,
        "opcode": OPCODE_RDMA_WRITE_WITH_IMM,
        "signaled": True,
        "imm_data": imm,
        "imm_be": imm_be,
        "wr_id": pack_wr_ids(slot, nreqs)[-1],
        "chain_len": nreqs + (1 if extra else 0),
    }
    if extra and nreqs > 1:
        plan["num_sge"] = 1
        plan["sizes_len"] = nreqs * SIZEOF_INT
        plan["remote_addr"] = cmpls_addr + slot * COMPLETION_RECORD_SIZE
        # degraded: copy the rkey table one element at a time.
        rk = []
        for d in range(len(cmpls_rkeys)):
            rk.append(cmpls_rkeys[d])
        plan["rkey_by_dev"] = rk
    else:
        plan["num_sge"] = 0
        plan["sizes_len"] = 0
        plan["remote_addr"] = 0
        plan["rkey_by_dev"] = []
    return plan


_WR_COLUMNS = ("qp_index", "dev_index", "req_index", "length", "num_sge",
               "lkey", "rkey", "laddr", "raddr", "wr_id")


def _wr_record(state, sizes, lkeys, rkeys, slot, nreqs, r, qp_index, dev_index,
               rem_dev_index):
    """One work-request record; mutates the per-request cursor state in place."""
    size = sizes[r]
    off = state["off:%d" % r]
    cur = state["cur:%d" % r]
    left = size - off
    length = left if left < cur else cur
    rec = {
        "qp_index": qp_index,
        "dev_index": dev_index,
        "req_index": r,
        "length": length,
        "num_sge": 1 if length else 0,
        "lkey": lkeys[r][dev_index] if length else 0,
        "rkey": rkeys[r][rem_dev_index],
        "laddr": state["laddr:%d" % r],
        "raddr": state["raddr:%d" % r],
        "wr_id": pack_wr_ids(slot, nreqs)[r],
    }
    state["cur:%d" % r] = length
    new_off = off + length
    state["off:%d" % r] = new_off if new_off < size else size
    state["laddr:%d" % r] = state["laddr:%d" % r] + length
    state["raddr:%d" % r] = state["raddr:%d" % r] + length
    return rec


def plan_multisend(sizes, data_addrs, lkeys, remote_addrs, rkeys, slot, nqps,
                   qp_dev, qp_remdev, req0_id):
    """Build the whole QP-major work-request plan for one CTS batch.

    For every request the length carried by successive queue pairs follows
    ``len_i = min(size - offset_i, len_{i-1})`` seeded with ``len_-1 =
    chunk_span(size, nqps)``, and ``offset_{i+1} = min(offset_i + len_i, size)``.
    Local and remote addresses advance by the length actually posted, so a
    request that runs dry keeps emitting zero-length (``num_sge == 0``) work
    requests on the remaining queue pairs -- the rkey is still selected because
    it is required even for a zero-byte write, while the lkey is left unset.

    Returns a columnar dict; every column has ``nqps * nreqs`` entries ordered
    ``for step in qp_ring(...): for r in range(nreqs)``.
    """
    nreqs = len(sizes)
    if nreqs < 1 or nreqs > NCCL_NET_IB_MAX_RECVS:
        raise ValueError("nreqs out of range 1..%d" % NCCL_NET_IB_MAX_RECVS)
    if not (len(data_addrs) == len(lkeys) == len(remote_addrs) == len(rkeys) == nreqs):
        raise ValueError("ragged per-request inputs")
    # degraded: string-keyed cursor state, one dict per work request, and the QP
    # ring plus the wr_id table are re-derived on every step.
    state = {}
    chunks = []
    for r in range(nreqs):
        if sizes[r] < 0:
            raise ValueError("negative send size")
        span = chunk_span(sizes[r], nqps)
        chunks.append(span)
        state["off:%d" % r] = 0
        state["cur:%d" % r] = span
        state["laddr:%d" % r] = data_addrs[r]
        state["raddr:%d" % r] = remote_addrs[r]

    cols = {}
    for name in _WR_COLUMNS:
        cols[name] = []
    for i in range(nqps):
        ring = qp_ring(nqps, req0_id, qp_dev, qp_remdev)
        qp_index, dev_index, rem_dev_index = ring[i]
        for r in range(nreqs):
            rec = _wr_record(state, sizes, lkeys, rkeys, slot, nreqs, r,
                             qp_index, dev_index, rem_dev_index)
            for name in _WR_COLUMNS:
                cols[name].append(rec[name])

    out = {
        "n": nqps * nreqs,
        "nqps": nqps,
        "nreqs": nreqs,
        "opcode": OPCODE_RDMA_WRITE,
        "chunk": chunks,
        "posted": [state["off:%d" % r] for r in range(nreqs)],
    }
    for name in _WR_COLUMNS:
        out[name] = cols[name]
    return out


def multisend_stats(plan):
    """Aggregate a plan: per-QP and per-request byte counts plus WR shape."""
    nqps = plan["nqps"]
    nreqs = plan["nreqs"]
    n = plan["n"]
    # degraded: one full rescan of the plan per queue pair and per request.
    per_qp = []
    for qp in range(nqps):
        per_qp.append(sum([plan["length"][j] for j in range(n)
                           if plan["qp_index"][j] == qp]))
    per_req = []
    for r in range(nreqs):
        per_req.append(sum([plan["length"][j] for j in range(n)
                            if plan["req_index"][j] == r]))
    zero_len = len([j for j in range(n) if plan["length"][j] == 0])
    sge_wrs = len([j for j in range(n) if plan["length"][j] != 0])
    max_len = 0
    for j in range(n):
        if plan["length"][j] > max_len:
            max_len = plan["length"][j]
    return {
        "bytes_per_qp": per_qp,
        "bytes_per_req": per_req,
        "total_bytes": sum(per_req),
        "zero_len_wrs": zero_len,
        "sge_wrs": sge_wrs,
        "max_len": max_len,
    }


def drain_cts_fifo(batches, nqps, qp_dev, qp_remdev, matching_scheme, ar,
                   ar_threshold, rem_ooo, local_ooo, cmpls_addr, cmpls_rkeys):
    """Drive the send path over a sequence of CTS batches.

    For each batch the posted sends are matched against the FIFO slots; once a
    multi-recv batch is fully matched the work-request plan, its trailing
    descriptor and its aggregate statistics are produced.  Batches that never
    complete are skipped, exactly like ``ncclIbIsend()`` returning without
    posting.
    """
    out = []
    for b in batches:
        slots = b["slots"]
        nreqs = slots[0]["nreqs"]
        taken = [False] * nreqs
        sizes = [0] * nreqs
        data = [0] * nreqs
        lks = [None] * nreqs
        matched = 0
        for p in b["posts"]:
            r, csize = match_send_slot(slots, b["idx"], p["tag"], p["size"], taken)
            if r < 0:
                continue
            taken[r] = True
            sizes[r] = csize
            data[r] = p["data"]
            lks[r] = p["lkeys"]
            matched += 1
        if matched < nreqs:
            continue
        plan = plan_multisend(
            sizes, data, lks,
            [slots[r]["addr"] for r in range(nreqs)],
            [slots[r]["rkeys"] for r in range(nreqs)],
            b["slot"], nqps, qp_dev, qp_remdev, b["req0_id"],
        )
        last = last_wr_plan(nreqs, b["req0_id"], sizes[0], matching_scheme, ar,
                            ar_threshold, rem_ooo, local_ooo, cmpls_addr,
                            b["slot"], cmpls_rkeys)
        out.append({"slot": b["slot"], "stats": multisend_stats(plan), "last": last})
    return out
