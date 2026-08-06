#!/usr/bin/env python3
"""Standalone verifier workload for the accepted-prefix KV commit subsystem
(scope: /app/repo/accept_compact.py :: plan_accept_move / commit_verified_step).
IMPLEMENT FROM AN EMPTY STUB.

Runs on CPU (numpy only, no torch, no GPU). Two modes:

  correctness : 39 curated batches that between them exercise every coupled contract
                point -- the ragged destination plan (all-zero accepted counts, fully
                accepted rows, mixed counts, sequence lengths of 0 and at the
                page-table limit, two requests sharing a page-table row, duplicated
                destinations so the later-plan-entry-wins rule is observable), the
                global flat survivor compaction (sentinels in the middle of a row, a
                row with fewer survivors than its accepted length and a row with more,
                an all-sentinel table), the sequence-length advance, the unfinished
                filter (empty, subset, unsorted, duplicated), the bonus-token index,
                the plan-only flavour and the accept_tokens / unfinished_index /
                kv_cache "None" paths, and non-contiguous inputs -- and compare every
                returned array against an INDEPENDENT reference computed here (NOT
                part of the editable scope).  Also asserts the 42 error contracts fire
                and that no input array is mutated.
  timing      : warmup + timed repeats of one large commit (bs=2048, width=16,
                dim=96) so the ragged plan, the compaction and the row movement all
                dominate.

Emits one line ``WRE_ACCEPT_RESULT {json}``.  Timing uses process_time (CPU time) so
the reward band is robust to OS descheduling under fleet load (exp 6.52).
"""
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

T_BS = 2048
T_WIDTH = 16
T_POOL_LEN = 256
T_ROWS = 2304
T_KV = 24576
T_DIM = 96
WARMUP = 1
ITERS = 3

DTYPES = {"tgt_cache_loc": np.int64, "accept_out_cache_loc": np.int64,
          "seq_lens_next": np.int64, "num_accept_tokens_filter": np.int64,
          "bonus_tokens": np.int64, "kv_cache": np.float32}
SCALARS = ("n_move", "n_accept")


def load_scope():
    import accept_compact as m
    return m


# --------------------------------------------------------------------------- #
# independent reference (definitional; NOT the editable scope)
# --------------------------------------------------------------------------- #
def _reference(rpi, r2t, seq_lens, ncd, ai, ocl, accept_tokens, unfinished, kv):
    bs = int(ai.shape[0])
    width = int(ai.shape[1])
    rpi_l = np.asarray(rpi).tolist()
    sl_l = np.asarray(seq_lens).tolist()
    acc_l = [int(v) + 1 for v in np.asarray(ncd).tolist()]
    ai_l = np.asarray(ai).tolist()
    ocl_l = np.asarray(ocl).tolist()
    r2t_l = np.asarray(r2t).tolist()

    tgt = [0] * (bs * width)
    cur = 0
    for j in range(bs):
        for t in range(acc_l[j]):
            tgt[cur] = int(r2t_l[rpi_l[j]][sl_l[j] + t])
            cur += 1
    n_move = cur
    aoc = [0] * (bs * width)
    n_acc = 0
    for j in range(bs):
        for v in ai_l[j]:
            if int(v) != -1:
                aoc[n_acc] = int(ocl_l[int(v)])
                n_acc += 1
    seq_next = np.asarray([sl_l[j] + acc_l[j] for j in range(bs)], dtype=np.int64)
    if unfinished is None:
        filt = None
    else:
        f = [0] * bs
        for v in np.asarray(unfinished).tolist():
            f[int(v)] = acc_l[int(v)]
        filt = np.asarray(f, dtype=np.int64)
    if accept_tokens is None:
        bonus = None
    else:
        at_l = np.asarray(accept_tokens).tolist()
        bonus = np.asarray([int(at_l[j][acc_l[j] - 1]) for j in range(bs)],
                           dtype=np.int64)
    if kv is None:
        kv_out = None
    else:
        src = np.asarray(kv)
        kv_out = np.array(src, dtype=np.float32)
        for p in range(n_move):
            kv_out[tgt[p]] = src[aoc[p]]
    return {"tgt_cache_loc": np.asarray(tgt, dtype=np.int64),
            "accept_out_cache_loc": np.asarray(aoc, dtype=np.int64),
            "n_move": int(n_move), "n_accept": int(n_acc),
            "seq_lens_next": seq_next, "num_accept_tokens_filter": filt,
            "bonus_tokens": bonus, "kv_cache": kv_out}


# --------------------------------------------------------------------------- #
# curated cases
# --------------------------------------------------------------------------- #
def _i(v):
    return np.asarray(v, dtype=np.int64)


def _f(v):
    return np.asarray(v, dtype=np.float32)


def _prefix_ai(bs, width, acc, base=0):
    a = np.full((bs, width), -1, dtype=np.int64)
    for j in range(bs):
        for t in range(int(acc[j])):
            a[j, t] = base + j * width + t
    return a


def _kv(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, d)).astype(np.float32)


def _cases():
    rng = np.random.default_rng(20260726)
    out = []

    def add(name, kind, rpi, r2t, sl, ncd, ai, ocl, at, unf, kv):
        out.append((name, kind, (rpi, r2t, sl, ncd, ai, ocl, at, unf, kv)))

    # --- small hand-built ---------------------------------------------------
    r2t = _i([[10, 11, 12, 13, 14, 15, 16, 17],
              [20, 21, 22, 23, 24, 25, 26, 27],
              [30, 31, 32, 33, 34, 35, 36, 37],
              [40, 41, 42, 43, 44, 45, 46, 47]])
    ocl8 = _i([50, 51, 52, 53, 54, 55, 56, 57])
    at2 = _i([[7, 8, 9, 10], [11, 12, 13, 14]])
    kv = _kv(64, 5, 1)

    add("plan_basic", "plan", _i([0, 2]), r2t, _i([0, 1]), _i([1, 2]),
        _prefix_ai(2, 4, [2, 3]), ocl8, None, None, None)
    add("commit_basic", "commit", _i([0, 2]), r2t, _i([0, 1]), _i([1, 2]),
        _prefix_ai(2, 4, [2, 3]), ocl8, at2, _i([0, 1]), kv)
    add("acc_all_zero", "commit", _i([1, 3]), r2t, _i([0, 0]), _i([0, 0]),
        _prefix_ai(2, 4, [1, 1]), ocl8, at2, _i([1]), kv)
    add("acc_full", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([3, 3]),
        _prefix_ai(2, 4, [4, 4]), ocl8, at2, _i([0, 1]), kv)
    add("bs1_w1", "commit", _i([2]), r2t, _i([3]), _i([0]),
        _i([[5]]), ocl8, _i([[99]]), _i([0]), kv)
    add("bs1_w1_plan", "plan", _i([2]), r2t, _i([3]), _i([0]),
        _i([[5]]), ocl8, None, None, None)
    add("sl_at_limit", "commit", _i([0, 1]), r2t, _i([6, 5]), _i([1, 2]),
        _prefix_ai(2, 4, [2, 3]), ocl8, at2, _i([0]), kv)
    add("shared_row", "commit", _i([1, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, _i([0, 1]), kv)
    dup = _i([[9, 9, 9, 9], [9, 9, 9, 9], [3, 4, 5, 6], [7, 8, 9, 10]])
    add("dup_dest", "commit", _i([0, 1]), dup, _i([0, 0]), _i([2, 2]),
        _prefix_ai(2, 4, [3, 3]), ocl8, at2, _i([0, 1]), kv)
    mid = _i([[100, -1, 101, -1], [-1, 102, -1, 103]])
    add("sentinel_mid", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        np.asarray([[0, -1, 2, -1], [-1, 5, -1, 7]], dtype=np.int64),
        ocl8, at2, _i([0, 1]), kv)
    add("fewer_survivors", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([3, 3]),
        np.asarray([[0, -1, -1, -1], [1, 2, -1, -1]], dtype=np.int64),
        ocl8, at2, _i([0, 1]), kv)
    add("more_survivors", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([0, 0]),
        np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64),
        ocl8, at2, _i([0, 1]), kv)
    add("all_sentinel", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        np.full((2, 4), -1, dtype=np.int64), ocl8, at2, _i([0, 1]), kv)
    add("unf_empty", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, _i([]), kv)
    add("unf_unsorted_dup", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 2]),
        _prefix_ai(2, 4, [2, 3]), ocl8, at2, _i([1, 0, 1]), kv)
    add("no_tokens", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, None, _i([0]), kv)
    add("no_unfinished", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, None, kv)
    add("no_kv", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, _i([0]), None)
    add("noncontig_r2t", "commit", _i([0, 1]), r2t[:, ::-1], _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, _i([0]), kv)
    add("noncontig_ai", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2])[:, ::-1], ocl8, at2, _i([0]), kv)
    add("noncontig_ocl", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8[::-1], at2, _i([0]), kv)
    add("noncontig_kv", "commit", _i([0, 1]), r2t, _i([0, 0]), _i([1, 1]),
        _prefix_ai(2, 4, [2, 2]), ocl8, at2, _i([0]), kv[:, ::-1])
    add("rpi_reversed", "commit", _i([3, 2, 1, 0]), r2t, _i([0, 1, 2, 3]),
        _i([1, 1, 1, 1]), _prefix_ai(4, 4, [2, 2, 2, 2]),
        _i(list(range(16))), _i([[1, 2, 3, 4]] * 4), _i([0, 3]), kv)

    # --- generated ----------------------------------------------------------
    for idx, (bs, width, pool_len, nrows, npool, dim) in enumerate(
            [(3, 4, 16, 5, 96, 3), (5, 2, 12, 6, 64, 4), (8, 6, 20, 9, 128, 2),
             (2, 8, 24, 3, 160, 6), (7, 3, 10, 8, 96, 5), (11, 4, 14, 12, 192, 3),
             (4, 5, 18, 4, 128, 7), (6, 7, 22, 7, 224, 2), (9, 2, 9, 10, 96, 4),
             (12, 3, 11, 13, 160, 3), (1, 6, 30, 2, 64, 8), (10, 4, 13, 11, 128, 5),
             (5, 5, 15, 5, 96, 6), (3, 7, 25, 4, 192, 3), (14, 2, 8, 15, 128, 4),
             (2, 3, 7, 2, 48, 9)]):
        ncd = rng.integers(0, width, size=bs, dtype=np.int64)
        sl = rng.integers(0, pool_len - width, size=bs, dtype=np.int64)
        rpi = rng.integers(0, nrows, size=bs, dtype=np.int64)
        r2tg = rng.integers(0, npool, size=(nrows, pool_len), dtype=np.int64)
        oclg = rng.integers(0, npool, size=bs * width, dtype=np.int64)
        aig = _prefix_ai(bs, width, ncd + 1)
        atg = rng.integers(0, 40000, size=(bs, width), dtype=np.int64)
        unf = np.asarray(sorted(rng.choice(bs, size=max(1, bs // 2), replace=False)),
                         dtype=np.int64)
        kvg = _kv(npool, dim, 100 + idx)
        kind = "plan" if idx % 4 == 3 else "commit"
        if kind == "plan":
            add("gen%d_plan" % idx, "plan", rpi, r2tg, sl, ncd, aig, oclg,
                None, None, None)
        else:
            add("gen%d" % idx, "commit", rpi, r2tg, sl, ncd, aig, oclg, atg, unf, kvg)
    return out


def _cmp(got, ref):
    if not isinstance(got, dict):
        return False, {"not_a_dict": True}
    detail = {}
    for k, want in ref.items():
        if k not in got:
            return False, {"missing": k}
        g = got[k]
        if k in SCALARS:
            detail[k] = bool(isinstance(g, (int, np.integer))
                             and not isinstance(g, bool) and int(g) == int(want))
            continue
        if want is None:
            detail[k] = bool(g is None)
            continue
        if g is None:
            detail[k] = False
            continue
        g = np.asarray(g)
        detail[k] = bool(g.shape == want.shape and g.dtype == DTYPES[k]
                         and np.array_equal(g, want))
    return all(detail.values()), detail


def _correctness_case(m):
    per_case = {}
    for name, kind, payload in _cases():
        arrs = [a for a in payload if isinstance(a, np.ndarray)]
        snap = [np.array(a, copy=True) for a in arrs]
        try:
            if kind == "plan":
                got = m.plan_accept_move(*payload[:6])
            else:
                got = m.commit_verified_step(*payload)
            ref = _reference(*payload)
        except NotImplementedError:
            return {"correctness_ok": False, "reason": "not_implemented"}
        except Exception as e:                                  # noqa: BLE001
            per_case[name] = False
            per_case[name + ":exc"] = type(e).__name__
            continue
        ok, detail = _cmp(got, ref)
        if not ok:
            per_case[name + ":detail"] = detail
        mut = all(np.array_equal(a, b) for a, b in zip(arrs, snap))
        if not mut:
            per_case[name + ":mutated"] = True
        per_case[name] = bool(ok and mut)

    errs = {}

    def _expect(tag, *a):
        try:
            m.commit_verified_step(*a)
        except ValueError:
            errs[tag] = True
        except Exception:                                       # noqa: BLE001
            errs[tag] = False
        else:
            errs[tag] = False

    r2t = _i([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    rpi = _i([0, 1])
    sl = _i([0, 0])
    ncd = _i([1, 1])
    ai = _prefix_ai(2, 4, [2, 2])
    ocl = _i([1, 2, 3, 4, 5, 6, 7, 8])
    at = _i([[1, 2, 3, 4], [5, 6, 7, 8]])
    unf = _i([0, 1])
    kv = _kv(40, 3, 7)
    G = (rpi, r2t, sl, ncd, ai, ocl, at, unf, kv)

    def sub(**kw):
        names = ("rpi", "r2t", "sl", "ncd", "ai", "ocl", "at", "unf", "kv")
        vals = list(G)
        for k, v in kw.items():
            vals[names.index(k)] = v
        return tuple(vals)

    _expect("ai_list", *sub(ai=ai.tolist()))
    _expect("ai_rank1", *sub(ai=ai.reshape(-1)))
    _expect("ai_dtype", *sub(ai=ai.astype(np.int32)))
    _expect("bs_zero", *sub(ai=np.zeros((0, 4), dtype=np.int64),
                            rpi=_i([]), sl=_i([]), ncd=_i([])))
    _expect("width_zero", *sub(ai=np.zeros((2, 0), dtype=np.int64)))
    _expect("bs_big", *sub(ai=np.zeros((m.MAX_BS + 1, 1), dtype=np.int64)))
    _expect("width_big", *sub(ai=np.zeros((2, m.MAX_WIDTH + 1), dtype=np.int64)))
    _expect("r2t_list", *sub(r2t=r2t.tolist()))
    _expect("r2t_rank1", *sub(r2t=r2t.reshape(-1)))
    _expect("r2t_dtype", *sub(r2t=r2t.astype(np.float32)))
    _expect("r2t_norows", *sub(r2t=np.zeros((0, 4), dtype=np.int64)))
    _expect("r2t_nocols", *sub(r2t=np.zeros((3, 0), dtype=np.int64)))
    _expect("rpi_dtype", *sub(rpi=rpi.astype(np.int32)))
    _expect("rpi_rank2", *sub(rpi=rpi.reshape(2, 1)))
    _expect("rpi_shape", *sub(rpi=_i([0, 1, 2])))
    _expect("sl_dtype", *sub(sl=sl.astype(np.int32)))
    _expect("sl_shape", *sub(sl=_i([0])))
    _expect("ncd_dtype", *sub(ncd=ncd.astype(np.int32)))
    _expect("ncd_shape", *sub(ncd=_i([1, 1, 1])))
    _expect("ocl_list", *sub(ocl=ocl.tolist()))
    _expect("ocl_rank2", *sub(ocl=ocl.reshape(2, 4)))
    _expect("ocl_dtype", *sub(ocl=ocl.astype(np.int32)))
    _expect("ocl_empty", *sub(ocl=np.zeros(0, dtype=np.int64)))
    _expect("rpi_neg", *sub(rpi=_i([-1, 1])))
    _expect("rpi_high", *sub(rpi=_i([0, 3])))
    _expect("sl_neg", *sub(sl=_i([-1, 0])))
    _expect("ncd_neg", *sub(ncd=_i([-1, 1])))
    _expect("acc_over_width", *sub(ncd=_i([4, 1])))
    _expect("slice_over_pool", *sub(sl=_i([3, 0])))
    _expect("ai_below_sentinel", *sub(ai=np.asarray([[0, -2, -1, -1], [1, 2, -1, -1]],
                                                   dtype=np.int64)))
    _expect("ai_high", *sub(ai=np.asarray([[0, 8, -1, -1], [1, 2, -1, -1]],
                                          dtype=np.int64)))
    _expect("at_dtype", *sub(at=at.astype(np.int32)))
    _expect("at_shape", *sub(at=_i([[1, 2], [3, 4]])))
    _expect("unf_dtype", *sub(unf=unf.astype(np.int32)))
    _expect("unf_rank2", *sub(unf=unf.reshape(2, 1)))
    _expect("unf_neg", *sub(unf=_i([-1])))
    _expect("unf_high", *sub(unf=_i([2])))
    _expect("kv_dtype", *sub(kv=kv.astype(np.float64)))
    _expect("kv_rank1", *sub(kv=kv.reshape(-1)))
    _expect("kv_norows", *sub(kv=np.zeros((0, 3), dtype=np.float32)))
    _expect("ocl_over_kv", *sub(kv=_kv(3, 3, 9)))
    _expect("dest_over_kv", *sub(kv=_kv(12, 3, 9)))

    names = [k for k in per_case if not k.endswith((":detail", ":exc", ":mutated"))]
    ok_all = all(bool(per_case[k]) for k in names)
    return {"correctness_ok": bool(ok_all and all(errs.values())),
            "n_cases": len(names), "n_errors": len(errs),
            "failed": [k for k in names if not per_case[k]],
            "failed_errors": [k for k, v in errs.items() if not v],
            "detail": {k: v for k, v in per_case.items() if k.endswith(":detail")}}


def _timing_inputs():
    rng = np.random.default_rng(20260101)
    ncd = rng.integers(0, T_WIDTH, size=T_BS, dtype=np.int64)
    sl = rng.integers(0, T_POOL_LEN - T_WIDTH, size=T_BS, dtype=np.int64)
    rpi = rng.integers(0, T_ROWS, size=T_BS, dtype=np.int64)
    r2t = rng.integers(0, T_KV, size=(T_ROWS, T_POOL_LEN), dtype=np.int64)
    ocl = rng.integers(0, T_KV, size=T_BS * T_WIDTH, dtype=np.int64)
    ai = _prefix_ai(T_BS, T_WIDTH, ncd + 1)
    at = rng.integers(0, 40000, size=(T_BS, T_WIDTH), dtype=np.int64)
    unf = np.arange(0, T_BS, 2, dtype=np.int64)
    kv = rng.random((T_KV, T_DIM)).astype(np.float32)
    return rpi, r2t, sl, ncd, ai, ocl, at, unf, kv


def _timing_case(m):
    args = _timing_inputs()

    def once():
        m.commit_verified_step(*args)

    try:
        for _ in range(WARMUP):
            once()
    except NotImplementedError:
        return -1.0
    ts = []
    for _ in range(ITERS):
        t0 = time.process_time()
        once()
        ts.append((time.process_time() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    m = load_scope()
    if mode == "correctness":
        res = _correctness_case(m)
        res["mode"] = "correctness"
        print("WRE_ACCEPT_RESULT " + json.dumps(res))
        sys.exit(0 if res.get("correctness_ok") else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        if ms < 0:
            print("WRE_ACCEPT_RESULT " + json.dumps({"mode": "timing",
                                                     "timing_ms": -1,
                                                     "reason": "not_implemented"}))
            sys.exit(3)
        print("WRE_ACCEPT_RESULT " + json.dumps({"mode": "timing", "timing_ms": ms,
                                                 "iters": ITERS, "bs": T_BS,
                                                 "width": T_WIDTH, "dim": T_DIM}))
        sys.exit(0)
    else:
        print("WRE_ACCEPT_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
