# Performance Optimization Task — submission entry point.
#
# Offline arena planner for the alloc/free stream. Strategy:
#   1. Build co-liveness neighbor lists (which blocks overlap in time).
#   2. Multi-start greedy: several orderings x {first-fit, best-fit} placement, keep best.
#   3. Local compaction passes.
#   4. Time-budgeted large-neighborhood search across several chains. Destroy operators:
#      height squeeze, gap-targeted window clear, peak-focused sample, vertical strip,
#      small random; reinsert with first/best-fit in varied orders; annealing acceptance
#      with plateau restarts. Two engines: numpy-vectorized placement (fast path) and a
#      pure-Python fallback.
#   5. Final compaction polish. Every placement avoids co-live occupied ranges, so the
#      plan is always valid.

import random
import time

try:
    import numpy as _np
    _HAVE_NP = True
except Exception:
    _np = None
    _HAVE_NP = False


def _colive_lists(N, alloc_step, free_step):
    cols = [[] for _ in range(N)]
    for b in range(N):
        ab, fb = alloc_step[b], free_step[b]
        for c in range(b + 1, N):
            if alloc_step[c] < fb and ab < free_step[c]:
                cols[b].append(c)
                cols[c].append(b)
    return cols


def _lowest_off(s, occ):
    occ.sort()
    off = 0
    for st, en in occ:
        if st >= off + s:
            break
        if en > off:
            off = en
    return off


def _bestfit_off(s, occ):
    occ.sort()
    best = None
    best_left = None
    prev_end = 0
    maxend = 0
    for st, en in occ:
        if en > maxend:
            maxend = en
        if st > prev_end:
            gap = st - prev_end
            if gap >= s and (best_left is None or gap - s < best_left):
                best = prev_end
                best_left = gap - s
        if en > prev_end:
            prev_end = en
    return best if best is not None else maxend


def _greedy(sizes, cols, order, rule):
    N = len(sizes)
    offsets = [0] * N
    placed = bytearray(N)
    fn = _bestfit_off if rule else _lowest_off
    for b in order:
        occ = [(offsets[c], offsets[c] + sizes[c]) for c in cols[b] if placed[c]]
        offsets[b] = fn(sizes[b], occ)
        placed[b] = 1
    return offsets


def _arena(sizes, offsets):
    return max(o + s for o, s in zip(offsets, sizes))


def _compact(sizes, cols, offsets, order, rule, max_passes=10):
    fn = _bestfit_off if rule else _lowest_off
    for _ in range(max_passes):
        improved = False
        for b in order:
            occ = [(offsets[c], offsets[c] + sizes[c]) for c in cols[b]]
            new = fn(sizes[b], occ)
            if new < offsets[b]:
                offsets[b] = new
                improved = True
        if not improved:
            break
    return offsets


def _lns_py(sizes, cols, init, t_end, rng):
    N = len(sizes)
    cur = init[:]
    ce = [cur[b] + sizes[b] for b in range(N)]
    cur_a = max(ce)
    best = cur[:]
    best_a = cur_a
    inD = bytearray(N)
    T = cur_a * 0.002
    since_imp = 0
    while time.time() < t_end:
        if since_imp > 800:
            cur = best[:]
            for b in range(N):
                ce[b] = cur[b] + sizes[b]
            cur_a = best_a
            since_imp = 0
            T = best_a * 0.001
        since_imp += 1
        m = ce.index(cur_a)
        r = rng.random()
        if r < 0.22:
            delta = max(1, int(cur_a * rng.uniform(0.002, 0.03)))
            thr = cur_a - delta
            D = [b for b in range(N) if ce[b] > thr]
            if len(D) > 70:
                D = [m]
        elif r < 0.44:
            sm = sizes[m]
            occ = sorted((cur[c], ce[c]) for c in cols[m])
            gaps = []
            prev = 0
            for st, en in occ:
                if st > prev:
                    gaps.append((st - prev, prev))
                if en > prev:
                    prev = en
            if gaps:
                gaps.sort(reverse=True)
                gi = 0 if rng.random() < 0.7 else rng.randrange(len(gaps))
                gw, gs = gaps[gi]
                if gw < sm:
                    win_end = gs + sm
                    D = [m] + [c for c in cols[m] if cur[c] < win_end and ce[c] > gs]
                else:
                    D = [m]
                for _ in range(rng.randint(0, 4)):
                    D.append(rng.randrange(N))
            else:
                D = [m]
        elif r < 0.68:
            D = [m]
            top_m = ce[m]
            below = [c for c in cols[m] if cur[c] < top_m]
            if below:
                k = rng.randint(4, min(28, len(below)))
                D.extend(rng.sample(below, k))
            for _ in range(rng.randint(0, 5)):
                D.append(rng.randrange(N))
        elif r < 0.85:
            x = rng.randint(0, max(1, cur_a - 1))
            w = rng.randint(1, max(1, cur_a // 6))
            D = [b for b in range(N) if cur[b] < x + w and x < ce[b]]
        else:
            D = [rng.randrange(N) for _ in range(rng.randint(3, 14))]
        D = list(set(D))
        if not D:
            continue
        saved = [(b, cur[b]) for b in D]
        for b in D:
            inD[b] = 1
        D.sort(key=lambda b: -sizes[b])
        rr = rng.random()
        if rr < 0.25:
            rng.shuffle(D)
        elif rr < 0.6 and m in D:
            D.remove(m)
            D.append(m)
        fn = _bestfit_off if rng.random() < 0.6 else _lowest_off
        for b in D:
            sb = sizes[b]
            occ = [(cur[c], ce[c]) for c in cols[b] if not inD[c]]
            cur[b] = fn(sb, occ)
            ce[b] = cur[b] + sb
            inD[b] = 0
        a2 = max(ce)
        d = a2 - cur_a
        if d <= 0 or rng.random() < pow(2.718281828, -d / max(T, 1)):
            cur_a = a2
            if a2 < best_a:
                best_a = a2
                best = cur[:]
                since_imp = 0
        else:
            for b, old in saved:
                cur[b] = old
                ce[b] = old + sizes[b]
        T *= 0.9995
    return best


class _NpState:
    __slots__ = ("sizes_np", "cols_np", "cur", "ce", "inD")

    def __init__(self, sizes, cols, init):
        self.sizes_np = _np.asarray(sizes, dtype=_np.int64)
        self.cols_np = [_np.asarray(c, dtype=_np.int64) for c in cols]
        self.cur = _np.asarray(init, dtype=_np.int64)
        self.ce = self.cur + self.sizes_np
        self.inD = _np.zeros(len(sizes), dtype=_np.uint8)

    def place(self, b, rule):
        nb = self.cols_np[b]
        if nb.size == 0:
            return 0
        sel = nb[self.inD[nb] == 0]
        if sel.size == 0:
            return 0
        s = self.sizes_np[b]
        st = self.cur[sel]
        en = self.ce[sel]
        order = _np.argsort(st)
        st = st[order]
        en = en[order]
        cm = _np.maximum.accumulate(en)
        gs = _np.empty(cm.size, dtype=_np.int64)
        gs[0] = 0
        if cm.size > 1:
            gs[1:] = cm[:-1]
        gaps = st - gs
        valid = gaps >= s
        if not valid.any():
            return int(cm[-1])
        if rule == 0:
            return int(gs[valid.argmax()])
        left = _np.where(valid, gaps, 1 << 62)
        return int(gs[left.argmin()])


def _lns_np(sizes, cols, init, t_end, rng):
    N = len(sizes)
    st_ = _NpState(sizes, cols, init)
    cur_np = st_.cur
    ce_np = st_.ce
    inD = st_.inD
    cols_np = st_.cols_np
    sizes_np = st_.sizes_np
    cur_a = int(ce_np.max())
    best = init[:]
    best_a = cur_a
    T = cur_a * 0.002
    since_imp = 0
    while time.time() < t_end:
        if since_imp > 800:
            cur_np[:] = best
            ce_np[:] = cur_np + sizes_np
            cur_a = best_a
            since_imp = 0
            T = best_a * 0.001
        since_imp += 1
        m = int(ce_np.argmax())
        cur_a = int(ce_np[m])
        r = rng.random()
        if r < 0.22:
            delta = max(1, int(cur_a * rng.uniform(0.002, 0.03)))
            thr = cur_a - delta
            D = _np.nonzero(ce_np > thr)[0].tolist()
            if len(D) > 70:
                D = [m]
        elif r < 0.44:
            sm = int(sizes_np[m])
            nb = cols_np[m]
            stv = cur_np[nb]
            env = ce_np[nb]
            order = _np.argsort(stv)
            stv = stv[order]
            env = env[order]
            gaps = []
            prev = 0
            for i in range(stv.size):
                s_i = int(stv[i])
                e_i = int(env[i])
                if s_i > prev:
                    gaps.append((s_i - prev, prev))
                if e_i > prev:
                    prev = e_i
            if gaps:
                gaps.sort(reverse=True)
                gi = 0 if rng.random() < 0.7 else rng.randrange(len(gaps))
                gw, gs = gaps[gi]
                if gw < sm:
                    win_end = gs + sm
                    D = [m] + [int(c) for c in nb if cur_np[c] < win_end and ce_np[c] > gs]
                else:
                    D = [m]
                for _ in range(rng.randint(0, 4)):
                    D.append(rng.randrange(N))
            else:
                D = [m]
        elif r < 0.68:
            D = [m]
            top_m = int(ce_np[m])
            nb = cols_np[m]
            below = nb[cur_np[nb] < top_m]
            if below.size:
                k = rng.randint(4, min(28, int(below.size)))
                D.extend(int(x) for x in rng.sample(below.tolist(), k))
            for _ in range(rng.randint(0, 5)):
                D.append(rng.randrange(N))
        elif r < 0.85:
            x = rng.randint(0, max(1, cur_a - 1))
            w = rng.randint(1, max(1, cur_a // 6))
            D = _np.nonzero((cur_np < x + w) & (x < ce_np))[0].tolist()
        else:
            D = [rng.randrange(N) for _ in range(rng.randint(3, 14))]
        D = list(set(D))
        if not D:
            continue
        saved = [(b, int(cur_np[b])) for b in D]
        inD[D] = 1
        D.sort(key=lambda b: -sizes[b])
        rr = rng.random()
        if rr < 0.25:
            rng.shuffle(D)
        elif rr < 0.6 and m in D:
            D.remove(m)
            D.append(m)
        rule = 1 if rng.random() < 0.6 else 0
        for b in D:
            off = st_.place(b, rule)
            cur_np[b] = off
            ce_np[b] = off + sizes_np[b]
            inD[b] = 0
        a2 = int(ce_np.max())
        d = a2 - cur_a
        if d <= 0 or rng.random() < pow(2.718281828, -d / max(T, 1)):
            cur_a = a2
            if a2 < best_a:
                best_a = a2
                best = cur_np.tolist()
                since_imp = 0
        else:
            for b, old in saved:
                cur_np[b] = old
                ce_np[b] = old + sizes_np[b]
        T *= 0.9995
    return best


def plan_arena(sizes, alloc_step, free_step):
    """Assign each block a byte offset in one arena for an ONLINE alloc/free stream.

    Blocks b and c whose live intervals [alloc_step, free_step) overlap must occupy
    disjoint byte ranges; freed bytes may be reused. Returns offsets minimizing the
    arena high-water mark max_b(offsets[b] + sizes[b]).
    """
    N = len(sizes)
    if N == 0:
        return []

    budget = min(8.0, max(0.05, N * 0.02))
    t0 = time.time()
    t_end = t0 + budget

    cols = _colive_lists(N, alloc_step, free_step)
    by_size = sorted(range(N), key=lambda b: (-sizes[b], alloc_step[b], b))
    by_time = sorted(range(N), key=lambda b: (alloc_step[b], -sizes[b], b))
    by_life = sorted(range(N), key=lambda b: (-(free_step[b] - alloc_step[b]), -sizes[b], b))
    by_sasc = sorted(range(N), key=lambda b: (sizes[b], alloc_step[b], b))

    best = None
    best_a = None

    def consider(off):
        nonlocal best, best_a
        A = _arena(sizes, off)
        if best_a is None or A < best_a:
            best_a = A
            best = off[:]

    for o in (by_size, by_time, by_life, by_sasc):
        for rule in (0, 1):
            consider(_greedy(sizes, cols, o, rule))
    consider(_compact(sizes, cols, best[:], by_size, 1))

    lns = _lns_np if _HAVE_NP else _lns_py
    nchains = 5
    for c in range(nchains):
        tl = min(t0 + budget * (c + 1) / nchains - 0.05, t_end - 0.05)
        if tl <= time.time():
            break
        cand = lns(sizes, cols, best, tl, random.Random(1000 + c))
        consider(cand)

    best = _compact(sizes, cols, best, by_size, 1)
    return best


def custom_kernel(data):
    """Entry point the verifier calls.

    data = (sizes, alloc_step, free_step, config) where config = {"N": int}. Already wired to
    call plan_arena and return the offsets list.
    """
    sizes, alloc_step, free_step, config = data
    return plan_arena(sizes, alloc_step, free_step)
