# Reviewer-only ORACLE (not baked into the image): optimal contiguous min-MAX partition (balance P ring-all-reduce
# chunks) by binary search on the bottleneck + a greedy feasibility check. Never baked. Used only to
# (a) calibrate the bottleneck constant on the authoring lane and (b) prove correctness (validity) + the headroom
# gradient. Grounded in TRAIN.PARALLEL.DECENTRALIZED ring all-reduce / reduce-scatter: a lock-step
# ring's time is bounded by the largest chunk, so minimize the max chunk's bytes with <= P contiguous
# chunks. Binary-search the answer C over [max(sizes), sum(sizes)]; a candidate C is feasible iff a
# left-to-right greedy (start a new chunk whenever adding the next tensor would exceed C) uses <= P
# chunks. This is provably optimal for contiguous min-max partition. O(N log(sum)).

def balance_chunks(sizes, num_chunks):
    n = len(sizes)
    if n == 0:
        return []
    P = min(num_chunks, n)
    lo, hi = max(sizes), sum(sizes)

    def chunks_needed(cap):
        used, cur = 1, 0
        for s in sizes:
            if cur + s <= cap:
                cur += s
            else:
                used += 1
                cur = s
        return used

    while lo < hi:
        mid = (lo + hi) // 2
        if chunks_needed(mid) <= P:
            hi = mid
        else:
            lo = mid + 1
    cap = lo
    # reconstruct boundaries with the optimal cap (greedy fill)
    bounds = []
    cur = 0
    start_ok = 0
    for i, s in enumerate(sizes):
        if cur + s <= cap:
            cur += s
        else:
            bounds.append(i)      # close previous chunk at i (exclusive)
            cur = s
    bounds.append(n)
    return bounds


def custom_kernel(data):
    sizes, num_chunks, config = data
    return balance_chunks(sizes, num_chunks)
