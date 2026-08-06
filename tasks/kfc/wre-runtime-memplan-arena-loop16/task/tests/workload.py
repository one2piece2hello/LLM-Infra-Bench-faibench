#!/usr/bin/env python3
"""Verifier workload for wre-runtime-memplan-arena.

Drives the solver's /app/repo/submission/kernel.py `custom_kernel` on hidden ONLINE memory-arena
workloads. This is a PURE HOST-LOGIC task (no torch / no CUDA): the "kernel" is an online arena
allocator that services a time-ordered alloc/free stream, assigning each block a byte offset in
one contiguous arena.

Metric = peak_bytes (the arena high-water the plan achieves) — LOWER is better. It is emitted in
the `timing_ms` field so the shared, metric-agnostic test.sh computes
    vs_oracle = oracle_ms / candidate_ms = oracle_arena / candidate_arena
(candidate arena smaller than the oracle's -> vs_oracle > 1; equal -> 1.0; the bump-only
baseline2 -> 0 < vs_oracle < 1). The arena is computed HERE from the returned offsets, so it
cannot be faked; a smaller arena requires a genuinely better VALID packing.

Modes:
  correctness -> anti-cache probe (two different workloads, both plans must be VALID) + all
                 hidden workloads validated (every block placed; co-live blocks never share
                 bytes; offsets >= 0). Emits WRE_RESULT {"mode":"correctness","correctness_ok":bool,...}
  timing      -> arena high-water of the plan on the primary (large) workload. Emits
                 WRE_RESULT {"mode":"timing","timing_ms":<arena_bytes>,...}
"""
import collections
import json
import math
import random
import secrets
import sys

KERNEL_PATH = "/app/repo/submission/kernel.py"
_MB = 1 << 20

# Primary workload for the metric (large; runtime-arena-like reuse potential).
PRIMARY = {"N": 400, "seed": 900}

# Hidden correctness workloads — size-1, odd/tiny, medium, large, non-power-of-two.
CORRECTNESS_GRAPHS = [
    {"N": 1, "seed": 101},
    {"N": 7, "seed": 102},
    {"N": 33, "seed": 103},
    {"N": 150, "seed": 104},
    {"N": 400, "seed": 105},
    {"N": 257, "seed": 106},   # non-power-of-two
]

# Spread workloads (arena must scale with the work; diagnostic only, not a hard gate).
SPREAD_GRAPHS = [
    {"N": 40, "seed": 201},
    {"N": 200, "seed": 202},
    {"N": 400, "seed": 203},
]

_PER_GRAPH_SPREAD_MIN = 3.0


def _gen_graph(n, seed):
    """Deterministic online alloc/free stream: a few long-lived blocks + many short-lived ones.
    Block b allocs at step b and frees at b+L (half-open [b, b+L)); persistent blocks free at the
    end. Returns (sizes, alloc_step, free_step, config)."""
    rng = random.Random(seed)
    sizes = [0] * n
    alloc_step = [0] * n
    free_step = [0] * n
    T = n                        # timeline length
    n_persist = max(1, n // 50)  # ~2% long-lived, live essentially the whole stream
    for b in range(n):
        alloc_step[b] = b
        if b < n_persist:
            sizes[b] = rng.randint(4 * _MB, 8 * _MB)
            free_step[b] = T                          # lives [b, T)
        else:
            sizes[b] = rng.randint(256 * 1024, 3 * _MB)
            L = rng.randint(20, 45)                    # lifetime in steps
            fb = min(T, b + L)
            if fb <= b:
                fb = b + 1                             # guarantee alloc_step < free_step
            free_step[b] = fb
    return sizes, alloc_step, free_step, {"N": n}


def _load_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_kernel", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def _coerce_offsets(offsets, n):
    if offsets is None:
        return None, "returned None"
    try:
        off = [int(x) for x in offsets]
    except Exception as exc:
        return None, f"offsets not int-coercible: {exc}"
    if len(off) != n:
        return None, f"offsets length {len(off)} != N {n}"
    return off, "ok"


def _validate(offsets, sizes, alloc_step, free_step):
    """Return (valid, arena, msg). Valid iff all offsets >= 0 and every pair of
    simultaneously-live blocks occupies disjoint byte ranges. Live intervals are half-open
    [alloc_step, free_step), so at a shared step a freeing block is gone before an allocating
    one arrives — the sweep processes frees before allocs at each step."""
    n = len(sizes)
    off, msg = _coerce_offsets(offsets, n)
    if off is None:
        return False, -1, msg
    for b in range(n):
        if off[b] < 0:
            return False, -1, f"offset[{b}]={off[b]} < 0"
    arena = 0
    for b in range(n):
        end = off[b] + sizes[b]
        if end > arena:
            arena = end
    add = collections.defaultdict(list)
    rem = collections.defaultdict(list)
    for b in range(n):
        add[alloc_step[b]].append(b)
        rem[free_step[b]].append(b)
    times = sorted(set(add.keys()) | set(rem.keys()))
    live = set()
    for t in times:
        for b in rem.get(t, ()):     # frees first (half-open [alloc, free))
            live.discard(b)
        for b in add.get(t, ()):
            live.add(b)
        if len(live) > 1:
            iv = sorted((off[b], off[b] + sizes[b], b) for b in live)
            for a in range(len(iv) - 1):
                if iv[a][1] > iv[a + 1][0]:
                    return (False, -1,
                            f"co-live overlap at step {t}: block {iv[a][2]} "
                            f"[{iv[a][0]},{iv[a][1]}) vs block {iv[a + 1][2]} "
                            f"[{iv[a + 1][0]},{iv[a + 1][1]})")
    return True, arena, "valid"


def _plan(custom_kernel, sizes, alloc_step, free_step, config):
    return custom_kernel((list(sizes), list(alloc_step), list(free_step), dict(config)))


def run_correctness(custom_kernel):
    # Stage 0 — anti-cache probe: two DIFFERENT workloads, both plans must be VALID.
    ga = _gen_graph(96, secrets.randbits(30) | 1)
    gb = _gen_graph(129, secrets.randbits(30) | 1)
    oa = _plan(custom_kernel, *ga)
    va, aa, ma = _validate(oa, ga[0], ga[1], ga[2])
    if not va:
        return False, f"cache-probe workload A invalid: {ma}"
    ob = _plan(custom_kernel, *gb)
    vb, ab, mb = _validate(ob, gb[0], gb[1], gb[2])
    if not vb:
        return False, f"cache-probe workload B invalid: {mb}"
    # Stage 1 — all hidden workloads must yield a valid plan.
    for spec in CORRECTNESS_GRAPHS:
        sizes, alloc_step, free_step, config = _gen_graph(spec["N"], spec["seed"])
        offsets = _plan(custom_kernel, sizes, alloc_step, free_step, config)
        valid, arena, msg = _validate(offsets, sizes, alloc_step, free_step)
        if not valid:
            return False, f"workload N={spec['N']} seed={spec['seed']}: {msg}"
    return True, "all correctness passed"


def _arena_of(custom_kernel, spec):
    sizes, alloc_step, free_step, config = _gen_graph(spec["N"], spec["seed"])
    offsets = _plan(custom_kernel, sizes, alloc_step, free_step, config)
    valid, arena, msg = _validate(offsets, sizes, alloc_step, free_step)
    return valid, arena, msg


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    try:
        custom_kernel = _load_kernel()
    except Exception as exc:
        print("WRE_RESULT " + json.dumps({"mode": mode, "correctness_ok": False,
              "error": f"load_failed: {type(exc).__name__}: {exc}"}))
        sys.exit(3)

    if mode == "correctness":
        try:
            ok, msg = run_correctness(custom_kernel)
        except NotImplementedError as exc:
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"not_implemented: {exc}"}))
            sys.exit(3)
        except Exception as exc:
            import traceback
            print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": False,
                  "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]}))
            sys.exit(3)
        print("WRE_RESULT " + json.dumps({"mode": "correctness", "correctness_ok": bool(ok), "detail": msg}))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        try:
            valid, arena, msg = _arena_of(custom_kernel, PRIMARY)
            spread = []
            for spec in SPREAD_GRAPHS:
                v2, a2, _ = _arena_of(custom_kernel, spec)
                spread.append(a2 if v2 else -1)
        except Exception as exc:
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1,
                  "error": f"{type(exc).__name__}: {exc}"}))
            sys.exit(3)
        if not valid:
            print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": -1, "error": f"invalid_plan: {msg}"}))
            sys.exit(4)
        spread_ratio = (max(spread) / min(spread)) if (spread and min(spread) > 0) else float("inf")
        flat_ok = spread_ratio >= _PER_GRAPH_SPREAD_MIN
        print("WRE_RESULT " + json.dumps({"mode": "timing", "timing_ms": float(arena),
              "per_graph_spread": spread_ratio, "flat_ok": bool(flat_ok), "stable_ok": True,
              "primary": PRIMARY}))
        valid_metric = (arena > 0) and math.isfinite(arena)  # flat_ok is a diagnostic, not a gate
        sys.exit(0 if valid_metric else 4)
    else:
        print("WRE_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
