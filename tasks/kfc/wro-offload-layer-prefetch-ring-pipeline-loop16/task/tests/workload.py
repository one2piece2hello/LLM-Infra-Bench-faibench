#!/usr/bin/env python3
"""Verifier workload for the layer-wise weight-prefetch ring scope.

correctness: 24 scenarios checked against an INDEPENDENT pure-python reference
             written here in the harness (never imported from /app/repo).
             Everything is integer, so every comparison is EXACT.
timing:      768 layers chunked at 16 KiB (~166k chunks) over a 192-slot ring
             with a 2-layer lookahead; CPU time via time.process_time (exp §6.52),
             min of 3, after a fixed 400 ms CPU warm-up that makes the measurement
             immune to the idle-clock ramp.
"""
import importlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app/repo")
TOKEN = "WRO_PREFETCHRING_RESULT"


def scope_module():
    return importlib.import_module("prefetch_ring")


# ---------------------------------------------------------------- reference
def r_chunks(lb, cb):
    c_layer, c_size, ptr = [], [], [0]
    for l, nb in enumerate(lb):
        rem = nb
        while rem > 0:
            take = cb if rem > cb else rem
            c_layer.append(l)
            c_size.append(take)
            rem -= take
        ptr.append(len(c_layer))
    return c_layer, c_size, ptr


def r_window(ptr, la, rs):
    L = len(ptr) - 1
    out = []
    for l in range(L):
        far = min(l + 1 + la, L)
        out.append(min(ptr[far], ptr[l] + rs))
    return out


def r_ring(cs, rs):
    slot, ev = [], []
    reuse = [0] * rs
    for i in range(len(cs)):
        s = i % rs
        slot.append(s)
        ev.append(i - rs if i >= rs else -1)
        reuse[s] += 1
    return slot, ev, reuse


def r_arr(cs, bw, fx, t0):
    start, done = [], []
    t = t0
    for v in cs:
        cost = (v + bw - 1) // bw + fx
        start.append(t)
        t += cost
        done.append(t)
    return start, done


def r_stall(ptr, cd, cu):
    L = len(ptr) - 1
    start, end, stall = [], [], []
    prev = 0
    for l in range(L):
        ready = 0
        for j in range(ptr[l], ptr[l + 1]):
            if cd[j] > ready:
                ready = cd[j]
        s = ready if ready > prev else prev
        start.append(s)
        stall.append(s - prev)
        prev = s + cu[l]
        end.append(prev)
    return start, end, stall


def r_pipe(lb, cb, rs, sb, la, bw, fx, cu, t0):
    c_layer, c_size, ptr = r_chunks(lb, cb)
    wnd = r_window(ptr, la, rs)
    slot, ev, reuse = r_ring(c_size, rs)
    s_c, d_c = r_arr(c_size, bw, fx, t0)
    l_s, l_e, l_st = r_stall(ptr, d_c, cu)
    C = len(c_size)
    L = len(lb)
    conf = []
    for i in range(C):
        if ev[i] >= 0 and s_c[i] < l_e[c_layer[ev[i]]]:
            conf.append(i)
    cost = [(v + bw - 1) // bw + fx for v in c_size]
    return {"n_chunks": C, "n_layers": L, "bytes_total": sum(lb),
            "total_us": l_e[L - 1] if L else 0,
            "stall_us": sum(l_st), "bubbles": sum(1 for v in l_st if v > 0),
            "conflicts": len(conf), "link_busy_us": sum(cost),
            "last_arrival_us": d_c[C - 1] if C else 0,
            "window": wnd, "layer_stall": l_st, "layer_start": l_s,
            "slot_reuse": reuse, "conflict_chunks": conf}


def eq(got, exp, tag):
    g = np.asarray(got, dtype=np.int64)
    e = np.asarray(exp, dtype=np.int64)
    if e.size == 0 and g.size == 0:
        return
    assert g.shape == e.shape, "%s shape %r != %r" % (tag, g.shape, e.shape)
    assert np.array_equal(g, e), "%s mismatch" % tag


# ---------------------------------------------------------------- scenarios
def mk_case(seed, L, kind):
    rng = np.random.default_rng(seed)
    lb = (rng.integers(4, 40, size=L).astype(np.int64) * 4096)
    cu = rng.integers(20, 200, size=L).astype(np.int64)
    if kind == "zero_layers":
        lb = np.zeros(L, dtype=np.int64)
    elif kind == "some_empty" and L >= 4:
        lb[1] = 0
        lb[L - 1] = 0
    elif kind == "compute_free":
        cu = np.zeros(L, dtype=np.int64)
    elif kind == "compute_heavy":
        cu = cu * 500
    elif kind == "lopsided" and L >= 3:
        lb[0] = 400 * 4096
    return lb, cu


CASES = [
    # name, L, kind, chunk, slots, slot_bytes, lookahead, bw, fixed, issue
    ("plain",         24, "plain",        16384, 12, 16384, 2, 512, 5, 0),
    ("some_empty",    24, "some_empty",   16384, 12, 16384, 2, 512, 5, 0),
    ("zero_layers",   12, "zero_layers",  16384,  8, 16384, 2, 512, 5, 0),
    ("compute_free",  20, "compute_free", 16384, 10, 16384, 2, 512, 5, 0),
    ("compute_heavy", 20, "compute_heavy", 16384, 10, 16384, 2, 512, 5, 0),
    ("lopsided",      18, "lopsided",     16384,  9, 16384, 2, 512, 5, 0),
    ("chunk_huge",    16, "plain",     1 << 24, 6, 1 << 24, 2, 512, 5, 0),
    ("chunk_tiny",     6, "plain",        4096,  8, 4096,   2, 512, 5, 0),
    ("slots_1",       16, "plain",       16384,  1, 16384,  2, 512, 5, 0),
    ("slots_huge",    16, "plain",       16384, 999, 16384, 2, 512, 5, 0),
    ("look_0",        20, "plain",       16384, 10, 16384,  0, 512, 5, 0),
    ("look_huge",     20, "plain",       16384, 10, 16384, 99, 512, 5, 0),
    ("bw_1",           8, "plain",       16384,  6, 16384,  2,   1, 5, 0),
    ("bw_huge",       20, "plain",       16384, 10, 16384,  2, 1 << 26, 5, 0),
    ("fixed_0",       20, "plain",       16384, 10, 16384,  2, 512, 0, 0),
    ("fixed_big",     20, "plain",       16384, 10, 16384,  2, 512, 900, 0),
    ("issue_late",    20, "plain",       16384, 10, 16384,  2, 512, 5, 7777),
    ("slot_slack",    18, "plain",       16384,  9, 1 << 20, 2, 512, 5, 0),
    ("one_layer",      1, "plain",       16384,  4, 16384,  2, 512, 5, 0),
    ("no_layers",      0, "plain",       16384,  4, 16384,  2, 512, 5, 0),
    ("deep",          64, "plain",        8192, 16, 8192,   3, 512, 5, 0),
    ("deep_tight",    64, "compute_free", 8192,  2, 8192,   1, 512, 5, 0),
    ("wide_chunks",   30, "lopsided",    32768, 20, 32768,  4, 256, 9, 3),
]


def do_correctness():
    m = scope_module()
    ok = 0
    bad = []
    try:
        lb, cu = mk_case(11, 14, "plain")
        cl, cs, ptr = m.layer_chunk_table(lb, 16384)
        rcl, rcs, rptr = r_chunks(lb.tolist(), 16384)
        eq(cl, rcl, "chunk_layer")
        eq(cs, rcs, "chunk_size")
        eq(ptr, rptr, "layer_ptr")
        assert np.asarray(cs).dtype == np.int64, "chunk dtype"
        e0 = m.layer_chunk_table(np.zeros(0, dtype=np.int64), 4096)
        assert np.asarray(e0[0]).shape == (0,) and np.asarray(e0[2]).tolist() == [0], \
            "empty chunk table"
        eq(m.layer_chunk_table(np.zeros(3, dtype=np.int64), 4096)[2], [0, 0, 0, 0],
           "all-empty ptr")
        for la, rs in ((0, 4), (2, 4), (99, 3), (1, 1), (2, 10 ** 6)):
            eq(m.prefetch_window(np.asarray(rptr, dtype=np.int64), la, rs),
               r_window(rptr, la, rs), "window la=%d rs=%d" % (la, rs))
        assert np.asarray(m.prefetch_window(np.zeros(1, dtype=np.int64), 2, 4)).shape \
            == (0,), "empty window"
        for rs in (1, 5, 4096):
            gs, ge, gr = m.ring_assign(np.asarray(rcs, dtype=np.int64), rs, 1 << 24)
            es, ee, er = r_ring(rcs, rs)
            eq(gs, es, "ring slot rs=%d" % rs)
            eq(ge, ee, "ring evict rs=%d" % rs)
            eq(gr, er, "ring reuse rs=%d" % rs)
        assert np.asarray(m.ring_assign(np.zeros(0, dtype=np.int64), 3,
                                        4096)[2]).shape == (3,), "empty ring reuse"
        for bw, fx, t0 in ((512, 5, 0), (1, 0, 0), (1 << 26, 900, 1234)):
            gs, gd = m.chunk_arrivals(np.asarray(rcs, dtype=np.int64), bw, fx, t0)
            es, ed = r_arr(rcs, bw, fx, t0)
            eq(gs, es, "arr start bw=%d" % bw)
            eq(gd, ed, "arr done bw=%d" % bw)
        assert np.asarray(m.chunk_arrivals(np.zeros(0, dtype=np.int64), 4, 0,
                                           0)[0]).shape == (0,), "empty arrivals"
        _s, dc = m.chunk_arrivals(np.asarray(rcs, dtype=np.int64), 512, 5, 0)
        for cvec in (cu, np.zeros(14, dtype=np.int64), cu * 400):
            gs, ge, gt = m.stall_profile(np.asarray(rptr, dtype=np.int64), dc, cvec)
            es, ee, et = r_stall(rptr, np.asarray(dc).tolist(), cvec.tolist())
            eq(gs, es, "stall start")
            eq(ge, ee, "stall end")
            eq(gt, et, "stall stall")
        z = m.stall_profile(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64),
                            np.zeros(0, dtype=np.int64))
        assert all(np.asarray(x).shape == (0,) for x in z), "empty stall profile"
        # error paths
        for f, args in ((m.layer_chunk_table, (lb.reshape(-1, 2), 4096)),
                        (m.layer_chunk_table, (lb, 0)),
                        (m.layer_chunk_table, (-lb - 1, 4096)),
                        (m.prefetch_window, (np.zeros(0, dtype=np.int64), 2, 4)),
                        (m.prefetch_window, (np.array([1, 2], dtype=np.int64), 2, 4)),
                        (m.prefetch_window, (np.array([0, 5, 3], dtype=np.int64), 2, 4)),
                        (m.prefetch_window, (np.array([0, 3], dtype=np.int64), -1, 4)),
                        (m.prefetch_window, (np.array([0, 3], dtype=np.int64), 2, 0)),
                        (m.ring_assign, (np.asarray(rcs).reshape(-1, 1), 4, 1 << 24)),
                        (m.ring_assign, (np.asarray(rcs), 0, 1 << 24)),
                        (m.ring_assign, (np.asarray(rcs), 4, 0)),
                        (m.ring_assign, (np.asarray(rcs), 4, 8)),
                        (m.chunk_arrivals, (np.asarray(rcs).reshape(-1, 1), 4, 0, 0)),
                        (m.chunk_arrivals, (-np.asarray(rcs) - 1, 4, 0, 0)),
                        (m.chunk_arrivals, (np.asarray(rcs), 0, 0, 0)),
                        (m.chunk_arrivals, (np.asarray(rcs), 4, -1, 0)),
                        (m.chunk_arrivals, (np.asarray(rcs), 4, 0, -1)),
                        (m.stall_profile, (np.zeros(0, dtype=np.int64), dc, cu)),
                        (m.stall_profile, (np.asarray(rptr, dtype=np.int64), dc,
                                           cu[:3])),
                        (m.stall_profile, (np.asarray(rptr, dtype=np.int64), dc,
                                           -cu - 1)),
                        (m.stall_profile, (np.asarray(rptr, dtype=np.int64),
                                           dc[:-1], cu)),
                        (m.run_pipeline, (lb, 0, 4, 1 << 24, 2, 512, 5, cu)),
                        (m.run_pipeline, (lb, 16384, 4, 8, 2, 512, 5, cu)),
                        (m.run_pipeline, (lb, 16384, 4, 1 << 24, 2, 512, 5, cu[:2]))):
            try:
                f(*args)
                raise AssertionError("no ValueError from %s" % f.__name__)
            except ValueError:
                pass
        # no input mutation
        klb, kcu = lb.copy(), cu.copy()
        m.run_pipeline(lb, 16384, 8, 16384, 2, 512, 5, cu)
        assert np.array_equal(lb, klb), "layer_bytes mutated"
        assert np.array_equal(cu, kcu), "compute_us mutated"
        ok += 1
    except Exception as exc:  # noqa: BLE001
        bad.append("units:%s" % exc)
    for (name, L, kind, cb, rs, sb, la, bw, fx, t0) in CASES:
        try:
            lb, cu = mk_case(abs(hash(name)) % 99991, L, kind)
            got = m.run_pipeline(lb, cb, rs, sb, la, bw, fx, cu, t0)
            exp = r_pipe(lb.tolist(), cb, rs, sb, la, bw, fx, cu.tolist(), t0)
            for k in ("n_chunks", "n_layers", "bytes_total", "total_us", "stall_us",
                      "bubbles", "conflicts", "link_busy_us", "last_arrival_us"):
                assert int(got[k]) == int(exp[k]), "%s %r != %r" % (k, got[k], exp[k])
            for k in ("window", "layer_stall", "layer_start", "slot_reuse",
                      "conflict_chunks"):
                eq(got[k], exp[k], k)
            assert np.asarray(got["layer_stall"]).dtype == np.int64, "stall dtype"
            ok += 1
        except Exception as exc:  # noqa: BLE001
            bad.append("%s:%s" % (name, exc))
    total = len(CASES) + 1
    print("%s %s" % (TOKEN, json.dumps({
        "correctness_ok": ok == total,
        "correctness_frac": round(ok / float(total), 6),
        "cases": total, "passed": ok, "failures": bad[:6]})))


def _warm_cpu(target_ms=400.0):
    """Burn a FIXED amount of CPU, independent of the candidate, before timing.

    An idle box parks its cores at the minimum clock, and the first few hundred
    milliseconds of real work are billed at that clock -- `time.process_time` was
    measured up to 25x too large for the first ~5 iterations of an otherwise
    steady 3 ms workload. Burning the same fixed budget in every mode puts the
    governor at full clock before anything that counts is measured.
    """
    x = np.arange(8192, dtype=np.int64)
    t0 = time.process_time()
    while (time.process_time() - t0) * 1000.0 < target_ms:
        for _ in range(40):
            x = (x * 1103515245 + 12345) % 2147483647
    return int(x[0])


def do_timing():
    m = scope_module()
    L = int(os.environ.get("WRO_LAYERS", "768"))
    cb = int(os.environ.get("WRO_CHUNK", "16384"))
    rs = int(os.environ.get("WRO_SLOTS", "192"))
    la = int(os.environ.get("WRO_LOOKAHEAD", "2"))
    bw = int(os.environ.get("WRO_BW", "24000"))
    rng = np.random.default_rng(20260726)
    lb = rng.integers(12, 96, size=L).astype(np.int64) * 65536
    lb[::17] = 0
    cu = rng.integers(120, 900, size=L).astype(np.int64)
    cu[3::11] *= 4

    _warm_cpu(float(os.environ.get("WRO_WARM_MS", "400")))
    m.run_pipeline(lb[:4], cb, rs, cb, la, bw, 7, cu[:4], 0)
    r = m.run_pipeline(lb, cb, rs, cb, la, bw, 7, cu, 0)
    best = None
    for _ in range(3):
        t0 = time.process_time()
        r = m.run_pipeline(lb, cb, rs, cb, la, bw, 7, cu, 0)
        dt = (time.process_time() - t0) * 1000.0
        best = dt if best is None else min(best, dt)
    print("%s %s" % (TOKEN, json.dumps({
        "timing_ms": round(best, 4), "layers": L, "chunk": cb, "slots": rs,
        "lookahead": la, "n_chunks": int(r["n_chunks"]),
        "total_us": int(r["total_us"]), "stall_us": int(r["stall_us"]),
        "bubbles": int(r["bubbles"]), "conflicts": int(r["conflicts"]),
        "link_busy_us": int(r["link_busy_us"]),
        "window_sum": int(np.asarray(r["window"], dtype=np.int64).sum()),
        "reuse_sum": int(np.asarray(r["slot_reuse"], dtype=np.int64).sum()),
        "start_sum": int(np.asarray(r["layer_start"], dtype=np.int64).sum())})))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if cmd == "correctness":
        do_correctness()
    elif cmd == "timing":
        do_timing()
    else:
        raise SystemExit("usage: workload.py {correctness|timing}")
