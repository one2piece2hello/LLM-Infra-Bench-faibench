#!/usr/bin/env python3
"""Standalone verifier workload for the cloud-catalog cost selection subsystem
(scope: /app/repo/catalog_select.py :: select_offerings / select_offerings_with_lookup).
IMPLEMENT-FROM-EMPTY task.

Runs on CPU (numpy only, no torch, no GPU).  Two modes:

  correctness : curated batches that between them exercise every coupled contract
                point -- what a row costs under each of the three billing modes
                (including a mode-1 request that must take a spot price even when it
                is DEARER than the on-demand one, and a row billable under neither),
                the floors met exactly and missed by one, the pinned cloud, the exact
                accelerator (kind, count) match in all four of its shapes, the price
                cap hit exactly and missed by one, the cheapest pick with the
                tie-break walked down each of its three further levels, the fallback
                ladder's cost-then-region order, its truncation at max_ladder
                (including zero) while req_n_regions still counts the untruncated
                total, the per-row feasibility and pick counters, requests with
                nothing feasible, an empty catalog, an empty request batch, the name
                lookup resolving to the SMALLEST matching row index / missing
                entirely / over an empty query batch, the lookup-free flavour, the
                extreme prices and shapes, and non-contiguous and negative-stride
                inputs -- and compare every returned array against an INDEPENDENT
                reference computed here (NOT part of the editable scope).  Also
                asserts the error contracts fire and that no input is mutated.
  timing      : warmup + timed repeats of one large batch so the feasibility sweep,
                the pick, the ladder and the name lookup all dominate.

Emits one line ``WRE_CAT_RESULT {json}``.  Timing uses process_time (CPU time) so the
reward band is robust to OS descheduling under fleet load (exp 6.52).
"""
import json
import statistics
import sys
import time

import numpy as np

REPO = "/app/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

T_ROWS = 20000              # catalog rows in the timing batch
T_REQ = 12000               # resource requests in the timing batch
T_CLOUDS = 6                # clouds in the timing catalog
T_REGIONS = 220             # regions in the timing catalog
T_TYPES = 1400              # distinct instance-type names
T_KINDS = 8                 # accelerator kinds in the timing catalog
T_QUERY = 30000             # catalog name lookups in the timing batch
T_LADDER = 12               # fallback ladder cap used for timing
WARMUP = 1
ITERS = 3

REQ_KEYS = ("req_pick", "req_price", "req_spot", "req_n_feasible", "req_n_regions")
LAD_KEYS = ("ladder_ptr", "ladder_region", "ladder_cost")
ROW_KEYS = ("row_n_feasible", "row_n_picked")
INT_KEYS = REQ_KEYS + LAD_KEYS + ROW_KEYS
OPT_KEYS = ("query_row",)
ALL_KEYS = INT_KEYS + OPT_KEYS


def load_scope():
    import importlib
    return importlib.import_module("catalog_select")


def _i(x):
    return np.asarray(x, dtype=np.int64)


def _pack(rows, reqs, max_ladder, queries):
    """rows: (cloud, region, type, vcpu, mem, acc, accn, price, spot) each.

    reqs: (cloud, vcpu, mem, acc, accn, mode, cap) each.
    queries: None, or a list of (cloud, type).
    """
    rc = [[] for _ in range(9)]
    for row in rows:
        for k in range(9):
            rc[k].append(row[k])
    qc = [[] for _ in range(7)]
    for req in reqs:
        for k in range(7):
            qc[k].append(req[k])
    out = tuple(_i(c) for c in rc) + tuple(_i(c) for c in qc) + (max_ladder,)
    if queries is None:
        return out + (None, None)
    return out + (_i([q[0] for q in queries]), _i([q[1] for q in queries]))


def _reference(rcl, rrg, rty, rvc, rmm, rac, ran, rpr, rsp, qcl, qvc, qmm, qac, qan,
               qmd, qcp, max_ladder, kcl, kty):
    """Literal transcription of the documented contract.  Independent of the scope."""
    n_rows = int(rcl.shape[0])
    n_req = int(qcl.shape[0])
    Lcl = rcl.tolist()
    Lrg = rrg.tolist()
    Lty = rty.tolist()
    Lvc = rvc.tolist()
    Lmm = rmm.tolist()
    Lac = rac.tolist()
    Lan = ran.tolist()
    Lpr = rpr.tolist()
    Lsp = rsp.tolist()
    Qcl = qcl.tolist()
    Qvc = qvc.tolist()
    Qmm = qmm.tolist()
    Qac = qac.tolist()
    Qan = qan.tolist()
    Qmd = qmd.tolist()
    Qcp = qcp.tolist()

    req_pick = [-1] * n_req
    req_price = [-1] * n_req
    req_spot = [-1] * n_req
    req_nf = [0] * n_req
    req_nr = [0] * n_req
    row_nf = [0] * n_rows
    row_np_ = [0] * n_rows
    lad_reg = []
    lad_cost = []
    lad_ptr = [0]
    for q in range(n_req):
        feas = []
        for r in range(n_rows):
            if Qcl[q] >= 0 and Qcl[q] != Lcl[r]:
                continue
            if Lvc[r] < Qvc[q] or Lmm[r] < Qmm[q]:
                continue
            if Qac[q] < 0:
                if Lac[r] >= 0:
                    continue
            else:
                if Lac[r] != Qac[q] or Lan[r] != Qan[q]:
                    continue
            md = Qmd[q]
            if md == 0:
                if Lpr[r] < 0:
                    continue
                cost = Lpr[r]
                spot = 0
            elif md == 2:
                if Lsp[r] < 0:
                    continue
                cost = Lsp[r]
                spot = 1
            else:
                if Lsp[r] >= 0:
                    cost = Lsp[r]
                    spot = 1
                elif Lpr[r] >= 0:
                    cost = Lpr[r]
                    spot = 0
                else:
                    continue
            if Qcp[q] >= 0 and cost > Qcp[q]:
                continue
            feas.append((cost, Lvc[r], Lmm[r], r, spot))
        req_nf[q] = len(feas)
        for f in feas:
            row_nf[f[3]] += 1
        if feas:
            feas.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            b = feas[0]
            req_pick[q] = b[3]
            req_price[q] = b[0]
            req_spot[q] = b[4]
            row_np_[b[3]] += 1
            best = {}
            for f in feas:
                g = Lrg[f[3]]
                if g not in best or f[0] < best[g]:
                    best[g] = f[0]
            req_nr[q] = len(best)
            items = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
            for g, v in items[:max_ladder]:
                lad_reg.append(g)
                lad_cost.append(v)
        lad_ptr.append(len(lad_reg))

    query_row = None
    if kcl is not None:
        first = {}
        for r in range(n_rows - 1, -1, -1):
            first[(Lcl[r], Lty[r])] = r
        query_row = _i([first.get((c, t), -1)
                        for c, t in zip(kcl.tolist(), kty.tolist())])
    return {"req_pick": _i(req_pick), "req_price": _i(req_price),
            "req_spot": _i(req_spot), "req_n_feasible": _i(req_nf),
            "req_n_regions": _i(req_nr), "ladder_ptr": _i(lad_ptr),
            "ladder_region": _i(lad_reg), "ladder_cost": _i(lad_cost),
            "row_n_feasible": _i(row_nf), "row_n_picked": _i(row_np_),
            "query_row": query_row}


def _cmp(got, exp):
    if not isinstance(got, dict):
        return False, "not a dict"
    if set(got.keys()) != set(ALL_KEYS):
        return False, "keys %r" % (sorted(got.keys()),)
    for k in INT_KEYS:
        g, e = got[k], exp[k]
        if not isinstance(g, np.ndarray):
            return False, k + ": not ndarray"
        if g.dtype != np.int64:
            return False, "%s: dtype %s" % (k, g.dtype)
        if g.ndim != 1:
            return False, "%s: ndim %d" % (k, g.ndim)
        if not g.flags["C_CONTIGUOUS"]:
            return False, k + ": not contiguous"
        if g.shape != e.shape:
            return False, "%s: shape %r vs %r" % (k, g.shape, e.shape)
        if g.size and not np.array_equal(g, e):
            bad = int(np.flatnonzero(g != e)[0])
            return False, "%s: [%d] %d vs %d" % (k, bad, int(g[bad]), int(e[bad]))
    g, e = got["query_row"], exp["query_row"]
    if e is None:
        if g is not None:
            return False, "query_row should be None"
    else:
        if not isinstance(g, np.ndarray) or g.dtype != np.int64 or g.ndim != 1:
            return False, "query_row: bad array"
        if not g.flags["C_CONTIGUOUS"]:
            return False, "query_row: not contiguous"
        if g.shape != e.shape:
            return False, "query_row: shape %r vs %r" % (g.shape, e.shape)
        if g.size and not np.array_equal(g, e):
            bad = int(np.flatnonzero(g != e)[0])
            return False, "query_row: [%d] %d vs %d" % (bad, int(g[bad]), int(e[bad]))
    return True, ""


# --- the shared demonstration catalog ---------------------------------------
# (cloud, region, type, vcpu, mem, acc, accn, price, spot)
CAT = [
    (0, 10, 100, 4, 16, -1, 0, 500, 300),        # 0  cheap spot
    (0, 10, 101, 8, 32, -1, 0, 900, -1),         # 1  no spot at all
    (0, 11, 100, 4, 16, -1, 0, 480, 620),        # 2  spot DEARER than on demand
    (0, 11, 102, 16, 64, -1, 0, 1700, 800),      # 3
    (1, 20, 100, 4, 16, -1, 0, 460, 340),        # 4  another cloud, same name
    (1, 20, 103, 4, 32, -1, 0, 700, -1),         # 5  same vcpu, more memory
    (1, 21, 104, 2, 8, -1, 0, -1, 220),          # 6  spot only
    (1, 21, 105, 4, 16, -1, 0, -1, -1),          # 7  billable under nothing
    (2, 30, 106, 8, 32, 0, 1, 3000, 1100),       # 8  one acc of kind 0
    (2, 30, 107, 16, 64, 0, 2, 5600, 2100),      # 9  two of kind 0
    (2, 31, 108, 16, 64, 1, 2, 6100, -1),        # 10 two of kind 1
    (2, 31, 106, 8, 32, 0, 1, 2900, -1),         # 11 dup (cloud, type) of row 8
    (3, 40, 109, 4, 16, -1, 0, 480, 300),        # 12 same cost as row 0, region 40
    (3, 41, 109, 4, 16, -1, 0, 500, 300),        # 13 ties row 0 on every key
    (3, 42, 110, 2, 16, -1, 0, 500, 300),        # 14 ties row 0 on cost, smaller vcpu
    (3, 43, 111, 4, 8, -1, 0, 500, 300),         # 15 ties row 0 on cost+vcpu, less mem
    (4, 50, 112, 96, 384, 2, 8, 41000, 12000),   # 16 eight of kind 2
    (4, 50, 113, 96, 384, 2, 4, 21000, -1),      # 17 four of kind 2
    (4, 51, 112, 96, 384, 2, 8, 39000, -1),      # 18 dup (cloud, type) of row 16
    (5, 60, 114, 1, 1, -1, 0, 0, 0),             # 19 free
]


def _cases():
    out = []

    def add(name, rows, reqs, max_ladder=8, queries=None):
        out.append((name, _pack(rows, reqs, max_ladder, queries)))

    A = [(-1, 0, 0, -1, 0, 0, -1)]               # a request that takes anything

    # --- degenerate shapes -------------------------------------------------
    add("empty_all", [], [])
    add("empty_all_lookup", [], [], queries=[])
    add("empty_catalog", [], A * 3)
    add("empty_catalog_lookup", [], A * 2, queries=[(0, 100), (3, 111)])
    add("empty_requests", CAT, [])
    add("empty_requests_lookup", CAT, [], queries=[(0, 100)])
    add("empty_query_batch", CAT, A * 2, queries=[])

    # --- billing modes -----------------------------------------------------
    add("mode0_only", CAT, [(-1, 0, 0, -1, 0, 0, -1)])
    add("mode1_any", CAT, [(-1, 0, 0, -1, 0, 1, -1)])
    add("mode2_only", CAT, [(-1, 0, 0, -1, 0, 2, -1)])
    add("mode1_spot_dearer", [CAT[2]], [(-1, 0, 0, -1, 0, 1, -1)])
    add("mode1_no_spot_falls_back", [CAT[1]], [(-1, 0, 0, -1, 0, 1, -1)])
    add("mode1_neither", [CAT[7]], [(-1, 0, 0, -1, 0, 1, -1)])
    add("mode0_no_price", [CAT[6]], [(-1, 0, 0, -1, 0, 0, -1)])
    add("mode2_no_spot", [CAT[1]], [(-1, 0, 0, -1, 0, 2, -1)])
    add("all_modes_same_group", CAT,
        [(-1, 0, 0, -1, 0, 0, -1), (-1, 0, 0, -1, 0, 1, -1),
         (-1, 0, 0, -1, 0, 2, -1), (-1, 4, 16, -1, 0, 1, -1),
         (-1, 4, 16, -1, 0, 0, -1)])

    # --- floors ------------------------------------------------------------
    add("vcpu_floor_exact", CAT, [(-1, 96, 0, 2, 8, 0, -1)])
    add("vcpu_floor_over", CAT, [(-1, 97, 0, 2, 8, 0, -1)])
    add("mem_floor_exact", CAT, [(-1, 0, 384, 2, 8, 0, -1)])
    add("mem_floor_over", CAT, [(-1, 0, 385, 2, 8, 0, -1)])
    add("both_floors", CAT, [(-1, 8, 32, -1, 0, 0, -1), (-1, 16, 32, -1, 0, 0, -1),
                             (-1, 8, 64, -1, 0, 0, -1)])
    add("zero_floors", CAT, [(-1, 0, 0, -1, 0, 1, -1)])

    # --- the pinned cloud --------------------------------------------------
    add("cloud_pinned", CAT, [(0, 0, 0, -1, 0, 0, -1), (1, 0, 0, -1, 0, 0, -1),
                              (5, 0, 0, -1, 0, 0, -1)])
    add("cloud_pinned_empty", CAT, [(2, 0, 0, -1, 0, 0, -1)])
    add("cloud_any_vs_pinned", CAT, [(-1, 4, 16, -1, 0, 0, -1),
                                     (3, 4, 16, -1, 0, 0, -1)])

    # --- the exact accelerator match ---------------------------------------
    add("acc_none_wanted", CAT, [(-1, 0, 0, -1, 0, 0, -1)])
    add("acc_kind_and_count", CAT, [(-1, 0, 0, 0, 1, 0, -1)])
    add("acc_count_mismatch", CAT, [(-1, 0, 0, 2, 2, 0, -1)])
    add("acc_kind_mismatch", CAT, [(-1, 0, 0, 1, 1, 0, -1)])
    add("acc_same_count_other_kind", CAT, [(-1, 0, 0, 0, 2, 0, -1),
                                           (-1, 0, 0, 1, 2, 0, -1)])
    add("acc_eight_not_four", CAT, [(-1, 0, 0, 2, 8, 0, -1),
                                    (-1, 0, 0, 2, 4, 0, -1)])
    add("acc_unknown_kind", CAT, [(-1, 0, 0, 7, 1, 0, -1)])

    # --- the price cap -----------------------------------------------------
    add("cap_exact", CAT, [(0, 4, 16, -1, 0, 0, 480)])
    add("cap_one_below", CAT, [(0, 4, 16, -1, 0, 0, 479)])
    add("cap_zero", CAT, [(-1, 0, 0, -1, 0, 0, 0)])
    add("cap_none", CAT, [(-1, 0, 0, -1, 0, 0, -1)])
    add("cap_bites_ladder", CAT, [(-1, 4, 16, -1, 0, 0, 500)])

    # --- the pick tie-break ------------------------------------------------
    add("tie_on_cost_only", [CAT[13], CAT[0]], [(-1, 4, 16, -1, 0, 0, -1)])
    add("tie_then_vcpu", [CAT[13], CAT[14]], [(-1, 0, 16, -1, 0, 0, -1)])
    add("tie_then_mem", [CAT[13], CAT[15]], [(-1, 4, 8, -1, 0, 0, -1)])
    add("tie_then_index", [CAT[13], CAT[13], CAT[13]], [(-1, 4, 16, -1, 0, 0, -1)])
    add("tie_full_ladder", CAT, [(3, 0, 0, -1, 0, 0, -1)])

    # --- the fallback ladder ----------------------------------------------
    add("ladder_cost_then_region", CAT, [(-1, 4, 16, -1, 0, 0, -1)])
    add("ladder_trunc_zero", CAT, [(-1, 4, 16, -1, 0, 0, -1)], max_ladder=0)
    add("ladder_trunc_one", CAT, [(-1, 4, 16, -1, 0, 0, -1)], max_ladder=1)
    add("ladder_trunc_two", CAT, [(-1, 4, 16, -1, 0, 0, -1)], max_ladder=2)
    add("ladder_trunc_over", CAT, [(-1, 4, 16, -1, 0, 0, -1)], max_ladder=64)
    add("ladder_mixed_batch", CAT,
        [(-1, 4, 16, -1, 0, 1, -1), (-1, 0, 0, 0, 1, 0, -1),
         (2, 0, 0, -1, 0, 0, -1), (-1, 0, 0, -1, 0, 2, -1),
         (-1, 96, 384, 2, 8, 1, -1)], max_ladder=3)
    add("ladder_one_region", [CAT[19]], [(-1, 0, 0, -1, 0, 0, -1)])

    # --- the row counters --------------------------------------------------
    add("row_counters", CAT,
        [(-1, 4, 16, -1, 0, 0, -1), (-1, 4, 16, -1, 0, 0, -1),
         (0, 4, 16, -1, 0, 1, -1), (-1, 0, 0, 0, 1, 2, -1),
         (-1, 0, 0, -1, 0, 2, 400)])
    add("nothing_feasible", CAT, [(-1, 1024, 0, -1, 0, 0, -1),
                                  (-1, 0, 0, 3, 4, 0, -1)])

    # --- the name lookup ---------------------------------------------------
    add("lookup_hits", CAT, A, queries=[(0, 100), (1, 100), (2, 106), (4, 112)])
    add("lookup_misses", CAT, A, queries=[(0, 999), (5, 100), (3, 106)])
    add("lookup_mixed", CAT, A,
        queries=[(2, 106), (2, 107), (0, 777), (5, 114), (4, 112), (4, 113)])
    add("lookup_repeat", CAT, A, queries=[(0, 100)] * 4)

    # --- the extremes ------------------------------------------------------
    ext = [(0, 0, 0, 1 << 20, 1 << 32, -1, 0, 1 << 40, 1 << 40),
           (0, 1, 1, 1 << 20, 1 << 32, 4095, 4096, 1 << 40, 0),
           (4095, 65535, (1 << 20) - 1, 1, 1, -1, 0, 0, -1)]
    add("extremes", ext, [(-1, 1 << 20, 1 << 32, -1, 0, 1, 1 << 40),
                          (-1, 0, 0, 4095, 4096, 2, -1),
                          (4095, 0, 0, -1, 0, 0, 0)],
        max_ladder=64, queries=[(4095, (1 << 20) - 1), (0, 1)])
    add("free_rows", [CAT[19], CAT[19]], [(-1, 0, 0, -1, 0, 1, 0)])

    # --- a wider batch so the grouping is really exercised -----------------
    wide_reqs = []
    for k in range(24):
        wide_reqs.append((-1 if k % 4 == 3 else k % 6, (k % 5) * 4, (k % 3) * 16,
                          -1 if k % 3 else (k % 3), 0 if k % 3 else (1 + k % 2),
                          k % 3, -1 if k % 5 else 3000))
    add("wide_batch", CAT, wide_reqs, max_ladder=4,
        queries=[(k % 6, 100 + (k % 15)) for k in range(9)])

    base = out[-1][1]
    bnq = out[-2][1]

    def _nc(a):
        if a is None or not isinstance(a, np.ndarray):
            return a
        buf = np.zeros((a.shape[0], 16), dtype=np.int64)
        buf[:, 5] = a
        v = buf[:, 5]
        assert not v.flags["C_CONTIGUOUS"] or a.shape[0] <= 1
        return v

    def _neg(a):
        if a is None or not isinstance(a, np.ndarray):
            return a
        v = a[::-1].copy()[::-1]
        assert v.strides[0] < 0
        return v

    out.append(("noncontig", tuple(_nc(a) for a in base)))
    out.append(("negstride", tuple(_neg(a) for a in base)))
    out.append(("noncontig_nolookup", tuple(_nc(a) for a in bnq)))
    return out


def _correctness_case(m):
    per_case = {}
    for name, args in _cases():
        snap = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        try:
            if args[17] is None and args[18] is None:
                got = m.select_offerings(*args[:17])
            else:
                got = m.select_offerings_with_lookup(*args)
        except Exception as exc:                                    # noqa: BLE001
            per_case[name] = False
            per_case[name + ":exc"] = "%s: %s" % (type(exc).__name__, exc)
            continue
        exp = _reference(*[np.ascontiguousarray(a) if isinstance(a, np.ndarray) else a
                           for a in args])
        ok, why = _cmp(got, exp)
        mutated = any(isinstance(a, np.ndarray) and not np.array_equal(a, b)
                      for a, b in zip(args, snap))
        per_case[name] = bool(ok and not mutated)
        if not ok:
            per_case[name + ":detail"] = why
        if mutated:
            per_case[name + ":mutated"] = True

    errs = {}

    def _expect(tag, *a):
        try:
            if a[17] is None and a[18] is None:
                m.select_offerings(*a[:17])
            else:
                m.select_offerings_with_lookup(*a)
        except ValueError:
            errs[tag] = True
        except Exception:                                           # noqa: BLE001
            errs[tag] = False
        else:
            errs[tag] = False

    G = (_i([0, 1, 2]), _i([10, 20, 30]), _i([100, 101, 102]), _i([4, 8, 16]),
         _i([16, 32, 64]), _i([-1, -1, 0]), _i([0, 0, 2]), _i([500, 900, 3000]),
         _i([300, -1, 1100]),
         _i([-1, 0]), _i([0, 4]), _i([0, 16]), _i([-1, -1]), _i([0, 0]),
         _i([0, 1]), _i([-1, 4000]), 8, _i([0, 2]), _i([100, 102]))
    NAMES = ("rcl", "rrg", "rty", "rvc", "rmm", "rac", "ran", "rpr", "rsp",
             "qcl", "qvc", "qmm", "qac", "qan", "qmd", "qcp", "ml", "kcl", "kty")

    def sub(**kw):
        vals = list(G)
        for k, v in kw.items():
            vals[NAMES.index(k)] = v
        return tuple(vals)

    for nm in NAMES[:16]:
        ref = G[NAMES.index(nm)]
        _expect(nm + "_list", *sub(**{nm: ref.tolist()}))
        _expect(nm + "_none", *sub(**{nm: None}))
        _expect(nm + "_int32", *sub(**{nm: ref.astype(np.int32)}))
        _expect(nm + "_float", *sub(**{nm: ref.astype(np.float64)}))
        _expect(nm + "_rank2", *sub(**{nm: ref.reshape(1, -1)}))
    for nm in NAMES[1:9]:
        _expect(nm + "_short", *sub(**{nm: G[NAMES.index(nm)][:2]}))
    for nm in NAMES[10:16]:
        _expect(nm + "_short", *sub(**{nm: G[NAMES.index(nm)][:1]}))
    _expect("rcl_negative", *sub(rcl=_i([-1, 1, 2])))
    _expect("rcl_too_big", *sub(rcl=_i([0, 1, m.MAX_CLOUDS])))
    _expect("rrg_negative", *sub(rrg=_i([-1, 20, 30])))
    _expect("rrg_too_big", *sub(rrg=_i([10, 20, m.MAX_REGIONS])))
    _expect("rty_negative", *sub(rty=_i([-1, 101, 102])))
    _expect("rty_too_big", *sub(rty=_i([100, 101, m.MAX_TYPES])))
    _expect("rvc_zero", *sub(rvc=_i([0, 8, 16])))
    _expect("rvc_negative", *sub(rvc=_i([-1, 8, 16])))
    _expect("rvc_too_big", *sub(rvc=_i([4, 8, m.MAX_VCPU + 1])))
    _expect("rmm_zero", *sub(rmm=_i([16, 0, 64])))
    _expect("rmm_too_big", *sub(rmm=_i([16, 32, m.MAX_MEM + 1])))
    _expect("rac_below", *sub(rac=_i([-2, -1, 0]), ran=_i([1, 0, 2])))
    _expect("rac_too_big", *sub(rac=_i([-1, -1, m.MAX_ACCS])))
    _expect("ran_negative", *sub(ran=_i([0, 0, -1])))
    _expect("ran_too_big", *sub(ran=_i([0, 0, m.MAX_ACC_COUNT + 1])))
    _expect("racn_kind_no_count", *sub(rac=_i([-1, -1, 0]), ran=_i([0, 0, 0])))
    _expect("racn_count_no_kind", *sub(rac=_i([-1, -1, -1]), ran=_i([0, 0, 2])))
    _expect("rpr_below", *sub(rpr=_i([-2, 900, 3000])))
    _expect("rpr_too_big", *sub(rpr=_i([500, 900, m.MAX_PRICE + 1])))
    _expect("rsp_below", *sub(rsp=_i([300, -2, 1100])))
    _expect("rsp_too_big", *sub(rsp=_i([300, -1, m.MAX_PRICE + 1])))
    _expect("qcl_below", *sub(qcl=_i([-2, 0])))
    _expect("qcl_too_big", *sub(qcl=_i([-1, m.MAX_CLOUDS])))
    _expect("qvc_negative", *sub(qvc=_i([-1, 4])))
    _expect("qvc_too_big", *sub(qvc=_i([0, m.MAX_VCPU + 1])))
    _expect("qmm_negative", *sub(qmm=_i([0, -1])))
    _expect("qmm_too_big", *sub(qmm=_i([0, m.MAX_MEM + 1])))
    _expect("qac_below", *sub(qac=_i([-2, -1])))
    _expect("qac_too_big", *sub(qac=_i([-1, m.MAX_ACCS]), qan=_i([0, 1])))
    _expect("qan_negative", *sub(qan=_i([0, -1])))
    _expect("qan_too_big", *sub(qac=_i([-1, 0]), qan=_i([0, m.MAX_ACC_COUNT + 1])))
    _expect("qacn_kind_no_count", *sub(qac=_i([-1, 0]), qan=_i([0, 0])))
    _expect("qacn_count_no_kind", *sub(qac=_i([-1, -1]), qan=_i([0, 2])))
    _expect("qmd_negative", *sub(qmd=_i([-1, 1])))
    _expect("qmd_too_big", *sub(qmd=_i([0, 3])))
    _expect("qcp_below", *sub(qcp=_i([-2, 4000])))
    _expect("qcp_too_big", *sub(qcp=_i([-1, m.MAX_PRICE + 1])))
    _expect("ml_float", *sub(ml=8.0))
    _expect("ml_bool", *sub(ml=True))
    _expect("ml_numpy", *sub(ml=np.int64(8)))
    _expect("ml_none", *sub(ml=None))
    _expect("ml_str", *sub(ml="8"))
    _expect("ml_negative", *sub(ml=-1))
    _expect("ml_too_big", *sub(ml=m.MAX_LADDER + 1))
    _expect("q_only_cloud", *sub(kty=None))
    _expect("q_only_type", *sub(kcl=None))
    for nm in ("kcl", "kty"):
        ref = G[NAMES.index(nm)]
        _expect(nm + "_list", *sub(**{nm: ref.tolist()}))
        _expect(nm + "_int32", *sub(**{nm: ref.astype(np.int32)}))
        _expect(nm + "_float", *sub(**{nm: ref.astype(np.float64)}))
        _expect(nm + "_rank2", *sub(**{nm: ref.reshape(1, -1)}))
    _expect("kcl_short", *sub(kcl=_i([0])))
    _expect("kcl_long", *sub(kcl=_i([0, 2, 1])))
    _expect("kcl_negative", *sub(kcl=_i([-1, 2])))
    _expect("kcl_too_big", *sub(kcl=_i([0, m.MAX_CLOUDS])))
    _expect("kty_negative", *sub(kty=_i([100, -1])))
    _expect("kty_too_big", *sub(kty=_i([100, m.MAX_TYPES])))
    wide = np.zeros(m.MAX_ROWS + 1, np.int64)
    _expect("rows_too_many", *sub(rcl=wide))
    del wide
    tall = np.zeros(m.MAX_REQUESTS + 1, np.int64)
    _expect("reqs_too_many", *sub(qcl=tall))
    del tall
    many = np.zeros(m.MAX_QUERIES + 1, np.int64)
    _expect("queries_too_many", *sub(kcl=many, kty=many))
    del many

    res = {"per_case": per_case, "errors": errs,
           "n_cases": sum(1 for k in per_case if ":" not in k),
           "n_errors": len(errs)}
    res["cases_ok"] = all(v for k, v in per_case.items() if ":" not in k)
    res["errors_ok"] = all(errs.values())
    res["correctness_ok"] = bool(res["cases_ok"] and res["errors_ok"])
    if not res["correctness_ok"]:
        res["failed"] = sorted(k for k, v in per_case.items()
                               if ":" not in k and not v)
        res["failed_errors"] = sorted(k for k, v in errs.items() if not v)
    return res


def _timing_inputs():
    rng = np.random.default_rng(20260726)
    n = T_ROWS
    # A catalog whose accelerator column looks like a real one: an accelerator catalog
    # is dominated by rows that DO carry one, spread over many (kind, count) pairs,
    # with a minority of plain CPU rows.
    row_cloud = rng.integers(0, T_CLOUDS, size=n).astype(np.int64)
    row_region = (row_cloud * (T_REGIONS // T_CLOUDS)
                  + rng.integers(0, T_REGIONS // T_CLOUDS, size=n)).astype(np.int64)
    row_type = rng.integers(0, T_TYPES, size=n).astype(np.int64)
    shape = rng.integers(0, 9, size=n).astype(np.int64)
    row_vcpu = (np.int64(2) << shape).astype(np.int64)
    row_mem = (row_vcpu * np.where(rng.random(n) < 0.5, 4, 8)).astype(np.int64)
    u = rng.random(n)
    kinds = np.arange(T_KINDS, dtype=np.int64)
    counts = np.array([1, 2, 4, 8, 16], dtype=np.int64)
    has = u >= 0.18
    ki = kinds[rng.integers(0, kinds.shape[0], size=n)]
    ci = counts[rng.integers(0, counts.shape[0], size=n)]
    row_acc = np.where(has, ki, -1).astype(np.int64)
    row_accn = np.where(has, ci, 0).astype(np.int64)
    base = (row_vcpu * 130 + row_mem * 9
            + np.where(has, row_accn * 4200, 0)).astype(np.int64)
    jit = rng.integers(80, 125, size=n).astype(np.int64)
    row_price = (base * jit // 100).astype(np.int64)
    row_price[rng.random(n) < 0.05] = -1
    sfrac = rng.integers(25, 75, size=n).astype(np.int64)
    row_spot = (row_price * sfrac // 100).astype(np.int64)
    row_spot[rng.random(n) < 0.25] = -1
    row_spot[row_price < 0] = np.where(rng.random(int((row_price < 0).sum())) < 0.5,
                                       -1, 700)

    q = T_REQ
    # Requests: most pin a cloud, all state floors, two thirds carry a price cap, and
    # the accelerator column mirrors the catalog's so every group is populated.
    v = rng.random(q)
    req_cloud = np.where(v < 0.72, rng.integers(0, T_CLOUDS, size=q), -1).astype(np.int64)
    qshape = rng.integers(2, 9, size=q).astype(np.int64)
    req_vcpu = (np.int64(2) << qshape).astype(np.int64)
    req_mem = (req_vcpu * np.where(rng.random(q) < 0.5, 4, 8)).astype(np.int64)
    w = rng.random(q)
    qhas = w >= 0.18
    qk = kinds[rng.integers(0, kinds.shape[0], size=q)]
    qc = counts[rng.integers(0, counts.shape[0], size=q)]
    req_acc = np.where(qhas, qk, -1).astype(np.int64)
    req_accn = np.where(qhas, qc, 0).astype(np.int64)
    req_mode = rng.integers(0, 3, size=q).astype(np.int64)
    want = (req_vcpu * 130 + req_mem * 9
            + np.where(qhas, req_accn * 4200, 0)).astype(np.int64)
    cf = rng.integers(60, 130, size=q).astype(np.int64)
    req_cap = (want * cf // 100).astype(np.int64)
    req_cap[rng.random(q) < 0.33] = -1

    query_cloud = rng.integers(0, T_CLOUDS, size=T_QUERY).astype(np.int64)
    query_type = rng.integers(0, T_TYPES, size=T_QUERY).astype(np.int64)
    return (np.ascontiguousarray(row_cloud), np.ascontiguousarray(row_region),
            np.ascontiguousarray(row_type), np.ascontiguousarray(row_vcpu),
            np.ascontiguousarray(row_mem), np.ascontiguousarray(row_acc),
            np.ascontiguousarray(row_accn), np.ascontiguousarray(row_price),
            np.ascontiguousarray(row_spot), np.ascontiguousarray(req_cloud),
            np.ascontiguousarray(req_vcpu), np.ascontiguousarray(req_mem),
            np.ascontiguousarray(req_acc), np.ascontiguousarray(req_accn),
            np.ascontiguousarray(req_mode), np.ascontiguousarray(req_cap),
            T_LADDER, np.ascontiguousarray(query_cloud),
            np.ascontiguousarray(query_type))


def _timing_case(m):
    args = _timing_inputs()

    def once():
        m.select_offerings_with_lookup(*args)

    for _ in range(WARMUP):
        once()
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
        print("WRE_CAT_RESULT " + json.dumps(res))
        sys.exit(0 if res.get("correctness_ok") else 3)
    elif mode == "timing":
        ms = _timing_case(m)
        print("WRE_CAT_RESULT " + json.dumps({"mode": "timing", "timing_ms": ms,
                                              "iters": ITERS, "n_rows": T_ROWS,
                                              "n_req": T_REQ, "n_query": T_QUERY}))
        sys.exit(0)
    else:
        print("WRE_CAT_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
