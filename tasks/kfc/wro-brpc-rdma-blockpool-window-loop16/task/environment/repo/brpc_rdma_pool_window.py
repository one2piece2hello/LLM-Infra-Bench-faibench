"""RDMA block-pool and credit-window planning for a brpc-style transport.

This module is the planning core that sits underneath an RDMA data path.  It
answers two questions that every zero-copy RDMA transport has to answer on the
hot path, and it answers them with plain integer arithmetic so that the answers
can be checked exactly:

  1. *Which registered memory region does this address belong to, and which
     block should I hand out for a payload of this size?*  RDMA hardware can
     only read from memory that has been registered with the NIC, and each
     registration yields an ``lkey`` that has to travel with every scatter/
     gather entry.  Registration is expensive, so a transport registers a small
     number of large ``Region`` slabs up front and then carves fixed-size
     blocks out of them.  Blocks come in three tiers -- 8 KiB, 64 KiB and
     2 MiB -- and each tier keeps its idle blocks in several independent
     buckets so that concurrent allocators do not all contend on one list.  On
     top of that each worker keeps a small thread-local cache of the smallest
     tier, refilled and drained in bulk.

  2. *Given a queue of outbound messages, what work requests should I post?*
     A reliable-connected queue pair can only have as many sends outstanding as
     the peer has receive work requests posted, so the sender maintains a
     credit window: it decrements a local counter for every work request it
     posts and the counter is replenished when the peer acknowledges.  The
     acknowledgement itself is piggybacked in the 32-bit immediate field of the
     next outbound send, which is why a send both consumes and carries credit.
     Two further batching policies ride along: completion events are only
     *solicited* from the peer once enough unsolicited traffic has accumulated,
     and local send completions are only *signalled* once every quarter window,
     because reaping a completion queue entry costs more than the send itself.

Both halves are exercised together: allocating a block gives you an address,
the address has to be resolved back to a region to get its ``lkey``, and the
resolved length is what the window loop cuts into scatter/gather entries.

Sizing
------
``BLOCK_SIZES`` and the reserved-work-request count are the real transport
constants and are not tunable here.  A pool is described by a bucket count and
a thread-local cache size; an endpoint is described by its own send/receive
queue depths plus the depths the peer advertised during the handshake.

Costing
-------
Every quantity in this module is an exact integer count of bytes or of work
requests -- there are no floats anywhere -- so a plan's totals do not depend on
the order in which a walk happens to accumulate them.

Implementation note
-------------------
This implementation is the SLOW-BUT-CORRECT reference path: it is what the
plans are checked against, and every routine here is written the most obvious
way rather than the fastest one.
"""

BYTES_IN_MB = 1048576

BLOCK_DEFAULT = 0
BLOCK_LARGE = 1
BLOCK_HUGE = 2
BLOCK_SIZE_COUNT = 3
BLOCK_SIZES = (8192, 65536, 2 * BYTES_IN_MB)

MIN_REGIONS = 1
MAX_REGIONS = 16

MIN_POOL_MB = 32
MAX_POOL_MB = 1048576

MIN_BUCKETS = 1
MAX_BUCKETS = 16

RESERVED_WR_NUM = 3
MIN_QP_SIZE = 16
MAX_QP_SIZE = 4096

ACK_FLUSH_SHIFT = 1
UNSOLICITED_BYTE_LIMIT = 1048576

TIER_NAMES = ("default", "large", "huge")
EVENT_NAMES = ("send", "recv")


def block_type_for(size):
    """Return the smallest block tier that can hold ``size`` bytes."""
    if not isinstance(size, int) or isinstance(size, bool):
        raise ValueError("size must be an int, got %r" % (size,))
    if size <= 0:
        raise ValueError("size must be positive, got %d" % size)
    if size > BLOCK_SIZES[BLOCK_SIZE_COUNT - 1]:
        raise ValueError("size %d exceeds largest block %d"
                         % (size, BLOCK_SIZES[BLOCK_SIZE_COUNT - 1]))
    for i in range(BLOCK_SIZE_COUNT):
        if size <= BLOCK_SIZES[i]:
            return i
    raise ValueError("unreachable: no tier for size %d" % size)


def regularize_region_size(size_mb, block_type, buckets):
    """Round a requested region size down to a whole number of blocks per bucket.

    A region is split evenly across ``buckets`` sub-slabs and every sub-slab
    must be a whole number of blocks, so the usable size is the request
    truncated to a multiple of ``block_size * buckets``.
    """
    if not isinstance(size_mb, int) or isinstance(size_mb, bool):
        raise ValueError("size_mb must be an int, got %r" % (size_mb,))
    if size_mb < MIN_POOL_MB or size_mb > MAX_POOL_MB:
        raise ValueError("size_mb %d outside [%d, %d]"
                         % (size_mb, MIN_POOL_MB, MAX_POOL_MB))
    if block_type < 0 or block_type >= BLOCK_SIZE_COUNT:
        raise ValueError("bad block_type %r" % (block_type,))
    if buckets < MIN_BUCKETS or buckets > MAX_BUCKETS:
        raise ValueError("buckets %r outside [%d, %d]"
                         % (buckets, MIN_BUCKETS, MAX_BUCKETS))
    raw = size_mb * BYTES_IN_MB
    step = BLOCK_SIZES[block_type] * buckets
    # SLOW: the usable size is just `raw - raw % step`, but this walks up to
    # SLOW: the answer one whole stripe at a time, so the cost grows with the
    # SLOW: number of blocks in the region instead of being constant.
    total = 0
    while total + step <= raw:
        total += step
    if total == 0:
        raise ValueError("region of %d MB holds no block of tier %d"
                         % (size_mb, block_type))
    return total


def region_table_new():
    """Create an empty registration table, kept sorted by region start."""
    return {"regions": [], "next_lkey": 1}


def region_table_add(table, start, size, block_type):
    """Register ``[start, start + size)`` as a new region and return its lkey.

    Regions may not overlap and the table is kept in ascending order of
    ``start`` so that an address can be resolved without scanning everything.
    """
    if not isinstance(table, dict) or "regions" not in table:
        raise ValueError("table must be a region table")
    regions = table["regions"]
    if len(regions) >= MAX_REGIONS:
        raise ValueError("region table already holds %d regions" % MAX_REGIONS)
    if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
        raise ValueError("start must be a positive int, got %r" % (start,))
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive int, got %r" % (size,))
    if start % 4096 != 0:
        raise ValueError("region start %d is not 4096-aligned" % start)
    if block_type < 0 or block_type >= BLOCK_SIZE_COUNT:
        raise ValueError("bad block_type %r" % (block_type,))
    if size % BLOCK_SIZES[block_type] != 0:
        raise ValueError("region size %d is not a multiple of block %d"
                         % (size, BLOCK_SIZES[block_type]))
    # SLOW: the table is already sorted, so a new region can only overlap the
    # SLOW: one region on either side of its insertion point.  This compares
    # SLOW: against every registered region instead.
    for other in regions:
        if start < other["start"] + other["size"] and other["start"] < start + size:
            raise ValueError("region [%d, %d) overlaps [%d, %d)"
                             % (start, start + size,
                                other["start"], other["start"] + other["size"]))
    lkey = table["next_lkey"]
    table["next_lkey"] = lkey + 1
    entry = {"start": start, "size": size, "block_type": block_type, "lkey": lkey}
    regions.append(entry)
    # SLOW: re-sorting the whole table by walking it back into place is an
    # SLOW: insertion sort; the insertion point could be found directly
    # SLOW: because the prefix is already ordered.
    i = len(regions) - 1
    while i > 0 and regions[i - 1]["start"] > regions[i]["start"]:
        regions[i - 1], regions[i] = regions[i], regions[i - 1]
        i -= 1
    return lkey


def region_of(table, addr):
    """Resolve ``addr`` to the region that contains it, or ``None``."""
    if not isinstance(table, dict) or "regions" not in table:
        raise ValueError("table must be a region table")
    if not isinstance(addr, int) or isinstance(addr, bool):
        raise ValueError("addr must be an int, got %r" % (addr,))
    # SLOW: the region list is sorted by start and the regions are disjoint,
    # SLOW: so the containing region can be found by bisecting on the starts.
    # SLOW: This walks every region on every single lookup, and this function
    # SLOW: is called at least once per block allocated, freed or posted.
    for entry in table["regions"]:
        if entry["start"] <= addr < entry["start"] + entry["size"]:
            return entry
    return None


def region_lkey(table, addr):
    """Return the registration key covering ``addr``, or 0 if unregistered."""
    entry = region_of(table, addr)
    if entry is None:
        return 0
    return entry["lkey"]


def bucket_index(table, addr, buckets):
    """Return which bucket of its region ``addr`` falls into."""
    if buckets < MIN_BUCKETS or buckets > MAX_BUCKETS:
        raise ValueError("buckets %r outside [%d, %d]"
                         % (buckets, MIN_BUCKETS, MAX_BUCKETS))
    entry = region_of(table, addr)
    if entry is None:
        raise ValueError("address %d is not in any region" % addr)
    # SLOW: `entry` above already holds the start and size; resolving the same
    # SLOW: address twice more doubles and triples the scan above.
    start = region_of(table, addr)["start"]
    size = region_of(table, addr)["size"]
    return ((addr - start) * buckets) // size


def _lcg_next(pool):
    """Advance the pool's deterministic bucket picker and return the new state."""
    state = (pool["rng"] * 1103515245 + 12345) & 0x7FFFFFFF
    pool["rng"] = state
    return state


def pool_create(buckets, tls_cache_num, base_addr=0x200000, seed=1):
    """Create an empty tiered block pool.

    ``buckets`` independent idle lists per tier spread allocator contention;
    ``tls_cache_num`` is how many smallest-tier blocks a worker may keep in its
    thread-local cache before it starts pushing them back to the shared lists.
    """
    if not isinstance(buckets, int) or isinstance(buckets, bool):
        raise ValueError("buckets must be an int, got %r" % (buckets,))
    if buckets < MIN_BUCKETS or buckets > MAX_BUCKETS:
        raise ValueError("buckets %d outside [%d, %d]"
                         % (buckets, MIN_BUCKETS, MAX_BUCKETS))
    if not isinstance(tls_cache_num, int) or isinstance(tls_cache_num, bool):
        raise ValueError("tls_cache_num must be an int, got %r" % (tls_cache_num,))
    if tls_cache_num < 2 or tls_cache_num > 4096:
        raise ValueError("tls_cache_num %d outside [2, 4096]" % tls_cache_num)
    if tls_cache_num % 2 != 0:
        raise ValueError("tls_cache_num %d must be even" % tls_cache_num)
    if not isinstance(base_addr, int) or base_addr <= 0 or base_addr % 4096 != 0:
        raise ValueError("base_addr must be a positive 4096-aligned int, got %r"
                         % (base_addr,))
    return {
        "buckets": buckets,
        "tls_cache_num": tls_cache_num,
        "idle": [[None] * buckets for _ in range(BLOCK_SIZE_COUNT)],
        "idle_size": [[0] * buckets for _ in range(BLOCK_SIZE_COUNT)],
        "expansion": [[None] * buckets for _ in range(BLOCK_SIZE_COUNT)],
        "expansion_size": [[0] * buckets for _ in range(BLOCK_SIZE_COUNT)],
        "region_num": [0] * BLOCK_SIZE_COUNT,
        "tls": None,
        "tls_num": 0,
        "rng": seed & 0x7FFFFFFF,
        "next_base": base_addr,
        "allocated": 0,
        "freed": 0,
        "extends": 0,
    }


def pool_extend(pool, table, size_mb, block_type):
    """Register one more region and stripe it across the tier's buckets.

    The new memory lands on the *expansion* list rather than the idle list; it
    is only promoted when an allocator actually finds its bucket empty, which
    keeps a burst of registrations from being handed straight back out.
    """
    if not isinstance(pool, dict) or "buckets" not in pool:
        raise ValueError("pool must be a block pool")
    if block_type < 0 or block_type >= BLOCK_SIZE_COUNT:
        raise ValueError("bad block_type %r" % (block_type,))
    buckets = pool["buckets"]
    size = regularize_region_size(size_mb, block_type, buckets)
    base = pool["next_base"]
    lkey = region_table_add(table, base, size, block_type)
    # Leave a guard gap and keep every region base aligned to the largest block
    # size, so that a block address is always aligned to its own tier.
    huge = BLOCK_SIZES[BLOCK_SIZE_COUNT - 1]
    nxt = base + size + huge
    pool["next_base"] = ((nxt + huge - 1) // huge) * huge
    stripe = size // buckets
    for i in range(buckets):
        node = {"start": base + i * stripe, "len": stripe,
                "next": pool["expansion"][block_type][i]}
        pool["expansion"][block_type][i] = node
        pool["expansion_size"][block_type][i] += stripe
    pool["region_num"][block_type] += 1
    pool["extends"] += 1
    return {"base": base, "size": size, "lkey": lkey, "stripe": stripe}


def _promote_expansion(pool, block_type, index):
    """Move a bucket's whole expansion list onto its (empty) idle list."""
    if pool["idle"][block_type][index] is not None:
        raise ValueError("idle list of tier %d bucket %d is not empty"
                         % (block_type, index))
    pool["idle"][block_type][index] = pool["expansion"][block_type][index]
    pool["idle_size"][block_type][index] += pool["expansion_size"][block_type][index]
    pool["expansion"][block_type][index] = None
    pool["expansion_size"][block_type][index] = 0


def _list_len(node):
    """Count the nodes on a chunk list."""
    n = 0
    while node is not None:
        n += 1
        node = node["next"]
    return n


def _list_bytes(node):
    """Sum the byte lengths on a chunk list."""
    total = 0
    while node is not None:
        total += node["len"]
        node = node["next"]
    return total


def pool_alloc(pool, table, size, grow_mb=MIN_POOL_MB):
    """Hand out one block big enough for ``size`` bytes and return its address.

    The smallest tier is served from the thread-local cache first; otherwise a
    bucket is picked, its head chunk is shaved by one block, and -- for the
    smallest tier only -- the rest of that bucket's whole blocks are pulled
    into the thread-local cache in bulk so the next few allocations are free.
    """
    block_type = block_type_for(size)
    bs = BLOCK_SIZES[block_type]
    if block_type == 0 and pool["tls"] is not None:
        node = pool["tls"]
        pool["tls"] = node["next"]
        # SLOW: the cache count is maintained exactly, so this is just
        # SLOW: `pool["tls_num"] -= 1`; recounting walks the whole cache.
        pool["tls_num"] = _list_len(pool["tls"])
        pool["allocated"] += 1
        return node["start"]

    index = _lcg_next(pool) % pool["buckets"]
    node = pool["idle"][block_type][index]
    if node is None:
        if pool["expansion"][block_type][index] is not None:
            _promote_expansion(pool, block_type, index)
            node = pool["idle"][block_type][index]
        if node is None:
            pool_extend(pool, table, grow_mb, block_type)
            _promote_expansion(pool, block_type, index)
            node = pool["idle"][block_type][index]
    if node is None:
        raise ValueError("tier %d bucket %d still empty after extend"
                         % (block_type, index))

    ptr = node["start"]
    if node["len"] > bs:
        node["start"] = node["start"] + bs
        node["len"] = node["len"] - bs
    else:
        pool["idle"][block_type][index] = node["next"]
    # SLOW: exactly `bs` bytes just left this bucket, so the running total
    # SLOW: could be decremented; instead the whole bucket is re-summed, which
    # SLOW: makes every allocation cost as much as the bucket is long.
    pool["idle_size"][block_type][index] = _list_bytes(
        pool["idle"][block_type][index])

    if block_type == 0:
        node = pool["idle"][0][index]
        pool["tls"] = node
        last_node = None
        while node is not None:
            if pool["tls_num"] > pool["tls_cache_num"] // 2 or node["len"] > bs:
                break
            # SLOW: one more node joined the cache, so the count is just
            # SLOW: `+= 1`; recounting inside the refill loop turns a linear
            # SLOW: bulk transfer into a quadratic one.
            last_node = node
            node = node["next"]
            pool["tls_num"] = _list_len(pool["tls"]) - _list_len(node)
        if pool["tls_num"] == 0:
            pool["tls"] = None
        else:
            pool["idle"][0][index] = node
        if last_node is not None:
            last_node["next"] = None
    pool["allocated"] += 1
    return ptr


def pool_recycle_tls(pool, table):
    """Flush the thread-local cache back to the shared buckets."""
    moved = 0
    while pool["tls"] is not None:
        node = pool["tls"]
        pool["tls"] = node["next"]
        entry = region_of(table, node["start"])
        if entry is None:
            continue
        index = ((node["start"] - entry["start"]) * pool["buckets"]) // entry["size"]
        node["next"] = pool["idle"][0][index]
        pool["idle"][0][index] = node
        pool["idle_size"][0][index] += node["len"]
        moved += 1
    pool["tls_num"] = 0
    return moved


def pool_dealloc(pool, table, addr):
    """Return the block at ``addr`` to the pool.

    Smallest-tier blocks go to the thread-local cache; once that is full half
    of it is spliced back onto the shared bucket in a single batch, which is
    what keeps the shared lists from being touched on every free.
    """
    entry = region_of(table, addr)
    if entry is None:
        raise ValueError("address %d is not in any region" % addr)
    block_type = entry["block_type"]
    bs = BLOCK_SIZES[block_type]
    if (addr - entry["start"]) % bs != 0:
        raise ValueError("address %d is not on a tier-%d block boundary"
                         % (addr, block_type))
    node = {"start": addr, "len": bs, "next": None}
    pool["freed"] += 1

    if block_type == 0 and pool["tls_num"] < pool["tls_cache_num"]:
        pool["tls_num"] += 1
        node["next"] = pool["tls"]
        pool["tls"] = node
        return 0

    index = ((addr - entry["start"]) * pool["buckets"]) // entry["size"]
    if block_type == 0:
        num = pool["tls_cache_num"] // 2
        # SLOW: the tail of the batch and its byte count both fall out of one
        # SLOW: walk of `num` nodes.  Restarting the walk from the head for
        # SLOW: every position makes the batch cost quadratic in its size.
        recycle_tail = None
        for i in range(num):
            step = pool["tls"]
            for _ in range(i):
                step = step["next"]
            recycle_tail = step
        length = 0
        for i in range(num):
            step = pool["tls"]
            for _ in range(i):
                step = step["next"]
            length += step["len"]
        new_head = pool["tls"]
        for _ in range(num):
            new_head = new_head["next"]
        if recycle_tail is not None:
            recycle_tail["next"] = node
            node["next"] = pool["idle"][0][index]
            pool["idle"][0][index] = pool["tls"]
            pool["idle_size"][0][index] += length + node["len"]
        pool["tls"] = new_head
        pool["tls_num"] -= num
    else:
        node["next"] = pool["idle"][block_type][index]
        pool["idle"][block_type][index] = node
        pool["idle_size"][block_type][index] += node["len"]
    return 0


def pool_stats(pool):
    """Summarise the pool: per-tier chunk counts, bytes and largest chunk."""
    if not isinstance(pool, dict) or "buckets" not in pool:
        raise ValueError("pool must be a block pool")
    tiers = []
    for bt in range(BLOCK_SIZE_COUNT):
        # SLOW: four separate walks of the same lists.  Every one of these
        # SLOW: numbers is available from a single pass over each bucket.
        chunks = 0
        for i in range(pool["buckets"]):
            chunks += _list_len(pool["idle"][bt][i])
        idle_bytes = 0
        for i in range(pool["buckets"]):
            idle_bytes += _list_bytes(pool["idle"][bt][i])
        exp_chunks = 0
        for i in range(pool["buckets"]):
            exp_chunks += _list_len(pool["expansion"][bt][i])
        exp_bytes = 0
        for i in range(pool["buckets"]):
            exp_bytes += _list_bytes(pool["expansion"][bt][i])
        max_chunk = 0
        for i in range(pool["buckets"]):
            step = pool["idle"][bt][i]
            while step is not None:
                if step["len"] > max_chunk:
                    max_chunk = step["len"]
                step = step["next"]
        tiers.append({
            "tier": TIER_NAMES[bt],
            "block_size": BLOCK_SIZES[bt],
            "chunks": chunks,
            "idle_bytes": idle_bytes,
            "expansion_chunks": exp_chunks,
            "expansion_bytes": exp_bytes,
            "max_chunk": max_chunk,
            "regions": pool["region_num"][bt],
        })
    return {
        "tiers": tiers,
        "tls_blocks": _list_len(pool["tls"]),
        "tls_num": pool["tls_num"],
        "allocated": pool["allocated"],
        "freed": pool["freed"],
        "extends": pool["extends"],
    }


def window_capacities(sq_size, rq_size, remote_sq_size, remote_rq_size):
    """Derive the two credit windows from the locally and remotely advertised depths.

    The sender may never have more work requests outstanding than either its
    own send queue or the peer's receive queue can hold, and a few slots are
    reserved so that a pure acknowledgement can always be posted.
    """
    depths = (sq_size, rq_size, remote_sq_size, remote_rq_size)
    for d in depths:
        if not isinstance(d, int) or isinstance(d, bool):
            raise ValueError("queue depths must be ints, got %r" % (depths,))
    clamped = []
    for d in depths:
        if d < MIN_QP_SIZE:
            d = MIN_QP_SIZE
        if d > MAX_QP_SIZE:
            d = MAX_QP_SIZE
        clamped.append(d)
    sq, rq, rsq, rrq = clamped
    local_cap = min(sq, rrq) - RESERVED_WR_NUM
    remote_cap = min(rq, rsq) - RESERVED_WR_NUM
    if local_cap < 1 or remote_cap < 1:
        raise ValueError("queue depths leave no window: local=%d remote=%d"
                         % (local_cap, remote_cap))
    return {"sq_size": sq, "rq_size": rq, "remote_sq_size": rsq,
            "remote_rq_size": rrq, "local_cap": local_cap,
            "remote_cap": remote_cap}


def endpoint_init(sq_size, rq_size, remote_sq_size, remote_rq_size,
                  remote_recv_block, max_sge):
    """Create the send-side state of a freshly handshaked queue pair."""
    if remote_recv_block not in BLOCK_SIZES:
        raise ValueError("remote_recv_block %r is not one of %r"
                         % (remote_recv_block, BLOCK_SIZES))
    if not isinstance(max_sge, int) or isinstance(max_sge, bool):
        raise ValueError("max_sge must be an int, got %r" % (max_sge,))
    if max_sge < 1 or max_sge > 64:
        raise ValueError("max_sge %d outside [1, 64]" % max_sge)
    caps = window_capacities(sq_size, rq_size, remote_sq_size, remote_rq_size)
    state = dict(caps)
    state.update({
        "remote_recv_block": remote_recv_block,
        "max_sge": max_sge,
        "remote_rq_window": caps["local_cap"],
        "sq_window": caps["local_cap"],
        "sq_imm_window": RESERVED_WR_NUM,
        "sq_current": 0,
        "sq_unsignaled": 0,
        "unsolicited": 0,
        "unsolicited_bytes": 0,
        "accumulated_ack": 0,
        "new_rq_wrs": 0,
        "posted": 0,
        "imm_sent": 0,
        "acks_flushed": 0,
    })
    return state


def _check_msgs(msgs):
    """Validate an outbound queue of (addr, len, block) descriptors."""
    if not isinstance(msgs, (list, tuple)) or len(msgs) == 0:
        raise ValueError("msgs must be a non-empty sequence")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise ValueError("msgs[%d] must be a dict, got %r" % (i, m))
        for key in ("addr", "len", "block"):
            if key not in m:
                raise ValueError("msgs[%d] is missing %r" % (i, key))
            if not isinstance(m[key], int) or isinstance(m[key], bool):
                raise ValueError("msgs[%d][%r] must be an int" % (i, key))
        if m["len"] <= 0:
            raise ValueError("msgs[%d] has non-positive len %d" % (i, m["len"]))
        if m["block"] not in BLOCK_SIZES:
            raise ValueError("msgs[%d] block %d is not a tier size"
                             % (i, m["block"]))
        if m["addr"] % m["block"] != 0:
            raise ValueError("msgs[%d] addr %d is not block-aligned"
                             % (i, m["addr"]))
        if m["len"] > m["block"]:
            raise ValueError("msgs[%d] len %d exceeds its block %d"
                             % (i, m["len"], m["block"]))


def plan_send_wrs(state, table, msgs):
    """Cut an outbound queue into the work requests a send would post.

    Each work request carries at most ``max_sge`` scatter/gather entries and at
    most one peer receive block worth of bytes; a scatter/gather entry never
    straddles a source block, since blocks are not contiguous.  Every request
    piggybacks the outstanding receive-credit count in its immediate field, and
    the solicited and signalled flags follow the batching policies described in
    the module docstring.  Runs out of window -> stops early (or reports
    ``eagain`` if nothing at all could be posted).
    """
    if not isinstance(state, dict) or "sq_window" not in state:
        raise ValueError("state must be an endpoint state")
    _check_msgs(msgs)
    ndata = len(msgs)
    recv_block = state["remote_recv_block"]
    max_sge = state["max_sge"]
    local_cap = state["local_cap"]
    remote_cap = state["remote_cap"]

    emitted = [[] for _ in range(ndata)]
    wrs = []
    total_len = 0
    current = 0
    rrw = state["remote_rq_window"]
    sqw = state["sq_window"]
    while current < ndata:
        if rrw == 0 or sqw == 0:
            if total_len > 0:
                break
            return {"wrs": wrs, "total_len": 0, "eagain": True,
                    "consumed": 0, "messages": ndata}
        slot = state["sq_current"]
        sges = []
        while True:
            # SLOW: both of these are running totals that the loop could carry
            # SLOW: forward, but they are rebuilt from the emitted lists on
            # SLOW: every scatter/gather entry, so filling one work request
            # SLOW: costs the square of the number of entries in it.
            this_len = 0
            for sge in sges:
                this_len += sge[1]
            if len(sges) >= max_sge or this_len >= recv_block:
                break
            off = 0
            for sge in emitted[current]:
                off += sge[1]
            if off >= msgs[current]["len"]:
                current += 1
                if current == ndata:
                    break
                continue
            m = msgs[current]
            room = m["block"] - (off % m["block"])
            take = m["len"] - off
            if room < take:
                take = room
            cap = recv_block - this_len
            if cap < take:
                take = cap
            addr = m["addr"] + off
            lkey = region_lkey(table, addr)
            if lkey == 0:
                raise ValueError("message %d address %d is not registered"
                                 % (current, addr))
            # SLOW: rebuilding the list for every entry copies everything that
            # SLOW: is already in it; appending in place does not.
            sges = sges + [(addr, take, lkey)]
            emitted[current] = emitted[current] + [(addr, take, lkey)]
        this_len = 0
        for sge in sges:
            this_len += sge[1]
        if this_len == 0:
            continue

        imm = state["new_rq_wrs"]
        state["new_rq_wrs"] = 0
        # SLOW: `current + 1 >= ndata` says the same thing without materialising
        # SLOW: a list of every message index still behind the cursor.
        tail = [i for i in range(ndata) if i > current]
        solicited = False
        if rrw == 1 or sqw == 1 or len(tail) == 0:
            solicited = True
        elif state["unsolicited"] > local_cap // 4:
            solicited = True
        elif state["accumulated_ack"] > remote_cap // 4:
            solicited = True
        elif state["unsolicited_bytes"] > UNSOLICITED_BYTE_LIMIT:
            solicited = True
        else:
            state["unsolicited"] += 1
            state["unsolicited_bytes"] += this_len
            state["accumulated_ack"] += imm
        if solicited:
            state["unsolicited"] = 0
            state["unsolicited_bytes"] = 0
            state["accumulated_ack"] = 0

        state["sq_unsignaled"] += 1
        signaled = False
        wr_id = 0
        if state["sq_unsignaled"] >= local_cap // 4:
            signaled = True
            wr_id = state["sq_unsignaled"]
            state["sq_unsignaled"] = 0

        wrs.append({"slot": slot, "num_sge": len(sges), "bytes": this_len,
                    "imm": imm, "solicited": solicited, "signaled": signaled,
                    "wr_id": wr_id, "sges": sges})
        total_len += this_len
        state["posted"] += 1
        state["sq_current"] += 1
        if state["sq_current"] == state["sq_size"] - RESERVED_WR_NUM:
            state["sq_current"] = 0
        rrw -= 1
        sqw -= 1
    state["remote_rq_window"] = rrw
    state["sq_window"] = sqw
    consumed = 0
    for lst in emitted:
        for sge in lst:
            consumed += sge[1]
    return {"wrs": wrs, "total_len": total_len, "eagain": False,
            "consumed": consumed, "messages": ndata}


def send_imm(state, imm):
    """Post a bare immediate-only work request carrying ``imm`` credits."""
    if not isinstance(imm, int) or isinstance(imm, bool) or imm < 0:
        raise ValueError("imm must be a non-negative int, got %r" % (imm,))
    if imm == 0:
        return 0
    if state["sq_imm_window"] <= 0:
        return 0
    state["sq_imm_window"] -= 1
    state["imm_sent"] += 1
    return 1


def send_ack(state, num):
    """Accumulate ``num`` reposted receive requests, flushing if past half window.

    Credits are normally piggybacked on the next data send; a standalone
    immediate is only spent once enough have piled up that the peer might
    otherwise stall.
    """
    if not isinstance(num, int) or isinstance(num, bool) or num < 0:
        raise ValueError("num must be a non-negative int, got %r" % (num,))
    old = state["new_rq_wrs"]
    state["new_rq_wrs"] = old + num
    if old > (state["remote_cap"] >> ACK_FLUSH_SHIFT) and state["sq_imm_window"] > 0:
        imm = state["new_rq_wrs"]
        state["new_rq_wrs"] = 0
        state["acks_flushed"] += 1
        return send_imm(state, imm)
    return 0


def handle_completion(state, table, events):
    """Reap completion events: return send credit, repost receives, count acks."""
    if not isinstance(events, (list, tuple)):
        raise ValueError("events must be a sequence")
    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or "kind" not in ev:
            raise ValueError("events[%d] must be a dict with 'kind'" % i)
        if ev["kind"] not in EVENT_NAMES:
            raise ValueError("events[%d] has bad kind %r" % (i, ev["kind"]))
    # SLOW: one pass per event kind over the same list; a single pass can bump
    # SLOW: both counters.
    n_send = 0
    for ev in events:
        if ev["kind"] == "send":
            n_send += 1
    n_recv = 0
    for ev in events:
        if ev["kind"] == "recv":
            n_recv += 1

    credits = 0
    reposted = 0
    acks = 0
    bad_lkey = 0
    seen_slots = []
    for ev in events:
        if ev["kind"] == "send":
            got = ev.get("wr_id", 0)
            if not isinstance(got, int) or isinstance(got, bool) or got < 0:
                raise ValueError("send event has bad wr_id %r" % (got,))
            room = state["local_cap"] - state["sq_window"]
            if got > room:
                got = room
            state["sq_window"] += got
            credits += got
            # SLOW: a set membership test is constant time; scanning the list
            # SLOW: of slots seen so far makes this quadratic in the batch.
            if ev.get("slot", -1) not in seen_slots:
                seen_slots = seen_slots + [ev.get("slot", -1)]
        else:
            imm = ev.get("imm", 0)
            if not isinstance(imm, int) or isinstance(imm, bool) or imm < 0:
                raise ValueError("recv event has bad imm %r" % (imm,))
            room = state["local_cap"] - state["remote_rq_window"]
            if imm > room:
                imm = room
            state["remote_rq_window"] += imm
            addr = ev.get("addr", 0)
            # SLOW: one lookup is enough -- the second and third calls repeat
            # SLOW: the whole region scan to learn nothing new.
            if region_lkey(table, addr) == 0:
                bad_lkey += 1
            elif region_of(table, addr) is None or region_lkey(table, addr) == 0:
                bad_lkey += 1
            else:
                reposted += 1
                acks += send_ack(state, 1)
    return {"send_events": n_send, "recv_events": n_recv,
            "credits_returned": credits, "reposted": reposted,
            "acks": acks, "bad_lkey": bad_lkey,
            "distinct_slots": len(seen_slots)}


def window_report(wrs):
    """Summarise a posted work-request plan."""
    if not isinstance(wrs, (list, tuple)):
        raise ValueError("wrs must be a sequence")
    if len(wrs) == 0:
        return {"wr_count": 0, "total_bytes": 0, "sge_count": 0, "max_sge": 0,
                "solicited": 0, "signaled": 0, "signal_bytes": 0,
                "imm_total": 0, "distinct_slots": 0, "max_bytes": 0,
                "first_slot": -1, "last_slot": -1}
    # SLOW: eight separate passes over the same plan.  Every one of these
    # SLOW: numbers can be folded in a single walk.
    total_bytes = 0
    for wr in wrs:
        total_bytes += wr["bytes"]
    sge_count = 0
    for wr in wrs:
        sge_count += wr["num_sge"]
    max_sge = 0
    for wr in wrs:
        if wr["num_sge"] > max_sge:
            max_sge = wr["num_sge"]
    solicited = 0
    for wr in wrs:
        if wr["solicited"]:
            solicited += 1
    signaled = 0
    signal_bytes = 0
    for wr in wrs:
        if wr["signaled"]:
            signaled += 1
    for wr in wrs:
        if wr["signaled"]:
            signal_bytes += wr["bytes"]
    imm_total = 0
    for wr in wrs:
        imm_total += wr["imm"]
    max_bytes = 0
    for wr in wrs:
        if wr["bytes"] > max_bytes:
            max_bytes = wr["bytes"]
    # SLOW: list membership again where a set would do.
    slots = []
    for wr in wrs:
        if wr["slot"] not in slots:
            slots = slots + [wr["slot"]]
    return {"wr_count": len(wrs), "total_bytes": total_bytes,
            "sge_count": sge_count, "max_sge": max_sge,
            "solicited": solicited, "signaled": signaled,
            "signal_bytes": signal_bytes, "imm_total": imm_total,
            "distinct_slots": len(slots), "max_bytes": max_bytes,
            "first_slot": wrs[0]["slot"], "last_slot": wrs[-1]["slot"]}


CFG_KEYS = ("buckets", "tls_cache_num", "pool_mb", "sq_size", "rq_size",
            "remote_sq_size", "remote_rq_size", "recv_block", "max_sge",
            "steps", "alloc_per_step", "msgs_per_step", "recv_per_step",
            "live_blocks", "max_alloc", "pre_regions")


def _check_cfg(cfg):
    """Validate a transfer configuration."""
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dict, got %r" % (cfg,))
    for key in CFG_KEYS:
        if key not in cfg:
            raise ValueError("cfg is missing %r" % (key,))
        v = cfg[key]
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("cfg[%r] must be an int, got %r" % (key, v))
    if cfg["steps"] < 1:
        raise ValueError("cfg['steps'] must be >= 1")
    for key in ("alloc_per_step", "msgs_per_step", "recv_per_step", "max_alloc"):
        if cfg[key] < 1:
            raise ValueError("cfg[%r] must be >= 1" % key)
    if cfg["live_blocks"] < cfg["alloc_per_step"]:
        raise ValueError("cfg['live_blocks'] must be >= cfg['alloc_per_step']")
    if cfg["max_alloc"] > BLOCK_SIZES[BLOCK_SIZE_COUNT - 1]:
        raise ValueError("cfg['max_alloc'] exceeds the largest block")
    if cfg["pre_regions"] < 0 or cfg["pre_regions"] > MAX_REGIONS - 1:
        raise ValueError("cfg['pre_regions'] %d outside [0, %d]"
                         % (cfg["pre_regions"], MAX_REGIONS - 1))


def run_transfer_tick(cfg, seed=1):
    """Drive one full allocate / post / reap / free cycle and digest the result.

    This is the entry point a transport's write path would call: it grows the
    registered pool on demand, cuts the outbound queue into work requests under
    the credit window, reaps the completions those requests would raise, and
    recycles the blocks that have fallen out of the live set.

    ``cfg["pre_regions"]`` registrations are made up front, round-robin over
    the tiers, which is what a process that pre-registers its whole pool at
    start-up looks like.
    """
    _check_cfg(cfg)
    table = region_table_new()
    pool = pool_create(cfg["buckets"], cfg["tls_cache_num"], seed=seed)
    for i in range(cfg["pre_regions"]):
        pool_extend(pool, table, cfg["pool_mb"], i % BLOCK_SIZE_COUNT)
    state = endpoint_init(cfg["sq_size"], cfg["rq_size"], cfg["remote_sq_size"],
                          cfg["remote_rq_size"], cfg["recv_block"],
                          cfg["max_sge"])
    rng = seed & 0x7FFFFFFF
    live = []
    digest = 0
    plans = 0
    eagains = 0
    total_bytes = 0
    total_sges = 0
    for _step in range(cfg["steps"]):
        for _ in range(cfg["alloc_per_step"]):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            size = 1 + (rng % cfg["max_alloc"])
            live.append((pool_alloc(pool, table, size, cfg["pool_mb"]), size))
        batch = live[-cfg["msgs_per_step"]:]
        msgs = []
        for (addr, size) in batch:
            bt = block_type_for(size)
            msgs.append({"addr": addr, "len": size, "block": BLOCK_SIZES[bt]})
        plan = plan_send_wrs(state, table, msgs)
        report = window_report(plan["wrs"])
        plans += 1
        if plan["eagain"]:
            eagains += 1
        total_bytes += report["total_bytes"]
        total_sges += report["sge_count"]

        events = [{"kind": "send", "wr_id": len(plan["wrs"]), "slot": 0}]
        for wr in plan["wrs"]:
            if wr["signaled"]:
                events.append({"kind": "send", "wr_id": 0, "slot": wr["slot"]})
        for (addr, _size) in live[-cfg["recv_per_step"]:]:
            events.append({"kind": "recv", "imm": 1, "addr": addr})
        events.append({"kind": "recv", "imm": len(plan["wrs"]),
                       "addr": live[0][0]})
        comp = handle_completion(state, table, events)

        while len(live) > cfg["live_blocks"]:
            addr, _size = live.pop(0)
            pool_dealloc(pool, table, addr)

        digest = (digest * 1000003
                  + report["total_bytes"]
                  + report["sge_count"] * 7
                  + report["solicited"] * 11
                  + report["signaled"] * 13
                  + report["distinct_slots"] * 17
                  + report["imm_total"] * 19
                  + comp["credits_returned"] * 23
                  + comp["reposted"] * 29
                  + comp["acks"] * 31
                  + comp["distinct_slots"] * 37
                  + plan["consumed"] * 41
                  + bucket_index(table, live[0][0], pool["buckets"]) * 43
                  ) & 0xFFFFFFFFFFFF
    before = pool_stats(pool)
    moved = pool_recycle_tls(pool, table)
    after = pool_stats(pool)
    return {"digest": digest, "plans": plans, "eagains": eagains,
            "total_bytes": total_bytes, "total_sges": total_sges,
            "posted": state["posted"], "imm_sent": state["imm_sent"],
            "acks_flushed": state["acks_flushed"],
            "sq_window": state["sq_window"],
            "remote_rq_window": state["remote_rq_window"],
            "new_rq_wrs": state["new_rq_wrs"],
            "sq_current": state["sq_current"],
            "regions": len(table["regions"]), "extends": pool["extends"],
            "allocated": pool["allocated"], "freed": pool["freed"],
            "tls_moved": moved, "live": len(live),
            "stats_before": before, "stats_after": after}


def transfer_sweep(cfgs, seed=1):
    """Run several configurations and fold their digests into one summary."""
    if not isinstance(cfgs, (list, tuple)) or len(cfgs) == 0:
        raise ValueError("cfgs must be a non-empty sequence")
    rows = []
    guard = 0
    bytes_total = 0
    for i, cfg in enumerate(cfgs):
        out = run_transfer_tick(cfg, seed=seed + i)
        rows.append(out)
        bytes_total += out["total_bytes"]
        guard = (guard * 31 + out["digest"] + out["posted"] * 7
                 + out["regions"] * 11 + out["extends"] * 13) & 0xFFFFFFFFFFFF
    return {"rows": rows, "guard": guard, "bytes_total": bytes_total,
            "count": len(rows)}
