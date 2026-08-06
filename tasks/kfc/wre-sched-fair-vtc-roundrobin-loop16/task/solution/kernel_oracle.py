import numpy as np


def fair_interleave_order(tenant_ids, num_tenants):
    t = np.asarray(tenant_ids, dtype=np.int64).ravel()
    N = t.shape[0]
    T = int(num_tenants)
    if N == 0:
        return np.empty(0, dtype=np.int64)

    counts = np.bincount(t, minlength=T)
    cmax = int(counts.max())
    if cmax == N:  # single active tenant -> identity schedule
        return np.arange(N, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    mx = cmax * T  # strictly above any (round * T + tenant) key
    dt = np.int32 if mx < 2**31 else np.int64

    if T * N < 2**31:
        # int32 composite sort key (tenant, arrival)
        ar = np.arange(N, dtype=np.int32)
        u = t.astype(np.int32) * N
        u += ar
        s = np.argsort(u)
        # k_s = round * T + tenant, fused: ar*T - repeat(starts*T - tenant, counts)
        k_s = ar
        k_s *= T
        adj = (starts * T - np.arange(T)).astype(np.int32)
        k_s -= np.repeat(adj, counts)
    else:
        ar32 = np.arange(N, dtype=np.int32)
        u = t * N
        u += ar32
        s = np.argsort(u)
        # overflow-safe build: round (= arange - group start) stays < cmax
        if dt is np.int32:
            r_s = ar32
        else:
            r_s = np.arange(N)
        r_s -= np.repeat(starts.astype(dt, copy=False), counts)
        r_s *= T
        r_s += np.repeat(np.arange(T, dtype=dt), counts)
        k_s = r_s

    if cmax * T <= 8 * N:
        # Q[r, t] = #{requests with round < r} + #{t' <= t: c_{t'} > r} - 1 = sched pos of (r, t)
        ind = np.arange(cmax, dtype=np.int64)[:, None] < counts[None, :]
        Q = np.cumsum(ind, axis=1, dtype=np.int32)
        h = np.bincount(counts, minlength=cmax + 1)
        ge = np.cumsum(h[::-1])[::-1][:cmax]
        A = np.cumsum(ge, dtype=np.int32) - ge[0]
        Q += (A - 1)[:, None]
        pos_s = Q.ravel()[k_s]
    elif mx <= 4 * (N + T) + 64:
        present = np.zeros(mx, dtype=np.bool_)
        present[k_s] = True
        tab = np.cumsum(present, dtype=np.int32)
        pos_s = tab[k_s]
        pos_s -= 1
    else:
        so = np.argsort(k_s)
        pos_s = np.empty(N, dtype=np.int32)
        pos_s[so] = np.arange(N, dtype=np.int32)

    out = np.empty(N, dtype=np.int32)
    out[s] = pos_s
    return out.astype(np.int64)


def custom_kernel(data):
    tenant_ids, num_tenants = data
    return fair_interleave_order(tenant_ids, num_tenants)
