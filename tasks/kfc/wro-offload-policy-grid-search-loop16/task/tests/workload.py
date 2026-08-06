#!/usr/bin/env python3
"""Verifier workload for wro-offload-policy-grid-search.

correctness: 22 hardware/model/budget scenarios checked against an INDEPENDENT
reference written here in the harness (never imported from /app/repo).
timing:      one full policy search on a steps=10 grid (287496 candidates);
             CPU time via time.process_time (exp §6.52).
"""
import importlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/app/repo")
TOKEN = "WRO_OFFPOLICY_RESULT"

GB = float(1 << 30)


def scope_module():
    return importlib.import_module("offload_policy")


# ---------------------------------------------------------------- reference
def ref_pairs(steps):
    return [(gi / steps, ci / steps)
            for gi in range(steps + 1) for ci in range(steps + 1 - gi)]


def ref_grid(steps):
    pr = ref_pairs(steps)
    return [(w[0], w[1], k[0], k[1], a[0], a[1])
            for w in pr for k in pr for a in pr]


def ref_search(sp, budgets, steps):
    grid = ref_grid(steps)
    n = int(sp["n_layers"])
    W = float(sp["weight_bytes_per_layer"])
    C = float(sp["kv_bytes_per_layer"])
    A = float(sp["act_bytes_per_layer"])
    pcie = float(sp["pcie_bytes_per_s"])
    disk = float(sp["disk_bytes_per_s"])
    tc = float(sp["flops_per_layer"]) / (float(sp["gpu_tflops"]) * 1e12)
    ws = float(sp["gpu_working_set_bytes"])
    pb = float(sp["pinned_buf_bytes"])
    over = float(sp["prefill_overhead_s"])
    best, best_lat, nf, bg, bc = -1, 0.0, 0, -1.0, -1.0
    for i, (wg, wc, cg, cc, hg, hc) in enumerate(grid):
        gpu = n * (wg * W + cg * C + hg * A) + ws
        cpu = n * (wc * W + cc * C + hc * A) + pb
        if not (gpu <= float(budgets[0]) and cpu <= float(budgets[1])):
            continue
        nf += 1
        t_w = wc * W / pcie + (1.0 - wg - wc) * W / disk
        t_k = cc * C / pcie + (1.0 - cg - cc) * C / disk
        t_a = hc * A / pcie + (1.0 - hg - hc) * A / disk
        s = t_w + t_k + t_a
        lat = n * (tc if tc > s else s) + over
        if best < 0 or lat < best_lat:
            best, best_lat, bg, bc = i, lat, gpu, cpu
    if best < 0:
        return {"best_index": -1, "best_policy": [], "best_latency": -1.0,
                "gpu_bytes": -1.0, "cpu_bytes": -1.0, "n_feasible": 0,
                "n_candidates": len(grid)}
    return {"best_index": best, "best_policy": [float(x) for x in grid[best]],
            "best_latency": float(best_lat), "gpu_bytes": float(bg),
            "cpu_bytes": float(bc), "n_feasible": nf, "n_candidates": len(grid)}


def norm(d):
    return {"best_index": int(d["best_index"]),
            "best_policy": [repr(float(x)) for x in d["best_policy"]],
            "best_latency": repr(float(d["best_latency"])),
            "gpu_bytes": repr(float(d["gpu_bytes"])),
            "cpu_bytes": repr(float(d["cpu_bytes"])),
            "n_feasible": int(d["n_feasible"]),
            "n_candidates": int(d["n_candidates"])}


# ---------------------------------------------------------------- scenarios
def specs(nl, wgb, kvgb, actmb, tflops, pcie_gbs, disk_gbs, wsgb, pinmb, over):
    return {"n_layers": nl,
            "weight_bytes_per_layer": wgb * GB / nl,
            "kv_bytes_per_layer": kvgb * GB / nl,
            "act_bytes_per_layer": actmb * (1 << 20),
            "gpu_tflops": tflops, "flops_per_layer": 2.4e11,
            "pcie_bytes_per_s": pcie_gbs * 1e9,
            "disk_bytes_per_s": disk_gbs * 1e9,
            "gpu_working_set_bytes": wsgb * GB,
            "pinned_buf_bytes": pinmb * (1 << 20),
            "prefill_overhead_s": over}


CASES = [
    ("opt6b_a100_loose", specs(32, 12, 4, 96, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 5),
    ("opt30b_a100_tight", specs(48, 60, 12, 192, 312, 16.0, 2.0, 2.0, 512, 0.75), (24 * GB, 64 * GB), 5),
    ("opt175b_nvme", specs(96, 325, 40, 384, 312, 16.0, 3.5, 3.0, 1024, 2.10), (40 * GB, 200 * GB), 5),
    ("tiny_gpu", specs(16, 8, 2, 48, 120, 8.0, 1.2, 1.0, 256, 0.12), (6 * GB, 32 * GB), 6),
    ("gpu_only_fits", specs(24, 5, 1, 32, 312, 16.0, 2.0, 0.5, 128, 0.08), (80 * GB, 512 * GB), 4),
    ("infeasible_gpu", specs(32, 400, 90, 512, 312, 16.0, 2.0, 60.0, 1024, 3.0), (8 * GB, 16 * GB), 4),
    ("slow_disk", specs(40, 70, 16, 160, 312, 16.0, 0.4, 2.0, 512, 0.9), (24 * GB, 128 * GB), 5),
    ("fast_disk", specs(40, 70, 16, 160, 312, 16.0, 12.0, 2.0, 512, 0.9), (24 * GB, 128 * GB), 5),
    ("slow_pcie", specs(40, 70, 16, 160, 312, 2.0, 2.0, 2.0, 512, 0.9), (24 * GB, 128 * GB), 5),
    ("compute_bound", specs(32, 6, 2, 32, 9, 24.0, 12.0, 1.0, 256, 0.05), (40 * GB, 256 * GB), 5),
    ("cpu_tight", specs(48, 60, 12, 192, 312, 16.0, 2.0, 2.0, 512, 0.75), (24 * GB, 20 * GB), 5),
    ("steps1", specs(32, 12, 4, 96, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 1),
    ("steps2", specs(32, 12, 4, 96, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 2),
    ("steps3", specs(32, 12, 4, 96, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 3),
    ("steps7", specs(20, 30, 8, 128, 312, 16.0, 2.0, 1.5, 512, 0.4), (16 * GB, 96 * GB), 7),
    ("huge_act", specs(32, 12, 4, 6144, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 4),
    ("zero_act", specs(32, 12, 4, 0, 312, 16.0, 2.0, 1.5, 512, 0.31), (40 * GB, 200 * GB), 4),
    ("one_layer", specs(1, 2, 1, 16, 312, 16.0, 2.0, 0.5, 64, 0.02), (12 * GB, 64 * GB), 5),
    ("exact_budget", specs(32, 12, 4, 96, 312, 16.0, 2.0, 1.5, 512, 0.31),
     (1.5 * GB + 12 * GB, 200 * GB), 4),
    ("both_tight", specs(64, 120, 24, 256, 312, 16.0, 2.0, 3.0, 768, 1.4), (20 * GB, 40 * GB), 6),
    ("disk_eq_pcie", specs(32, 24, 6, 96, 312, 4.0, 4.0, 1.5, 512, 0.31), (16 * GB, 96 * GB), 5),
    ("big_grid", specs(32, 24, 6, 96, 312, 16.0, 2.0, 1.5, 512, 0.31), (16 * GB, 96 * GB), 8),
]


def do_correctness():
    m = scope_module()
    ok = 0
    bad = []
    for name, sp, bud, steps in CASES:
        try:
            got = norm(m.search_best_policy(sp, bud, steps))
            exp = norm(ref_search(sp, bud, steps))
            assert got == exp, "search"
            g = m.enumerate_grid(steps)
            rg = np.array(ref_grid(steps), dtype=np.float64)
            assert g.shape == rg.shape and np.array_equal(g, rg), "grid"
            fm = m.feasible_mask(sp, g, bud)
            gpu, cpu = m.policy_peak_memory(sp, g)
            assert np.array_equal(
                np.asarray(fm, dtype=bool),
                (np.asarray(gpu) <= bud[0]) & (np.asarray(cpu) <= bud[1])), "mask"
            lat = np.asarray(m.policy_latency(sp, g))
            assert lat.shape == (rg.shape[0],), "lat_shape"
            if exp["best_index"] >= 0:
                assert repr(float(lat[int(exp["best_index"])])) == exp["best_latency"], "lat_val"
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad.append("%s:%s" % (name, e))
    print("%s %s" % (TOKEN, json.dumps({
        "correctness_ok": ok == len(CASES),
        "correctness_frac": round(ok / float(len(CASES)), 6),
        "cases": len(CASES), "passed": ok, "failures": bad[:6]})))


def do_timing():
    m = scope_module()
    steps = int(os.environ.get("WRO_STEPS", "10"))
    sp = specs(48, 60, 12, 192, 312, 16.0, 2.0, 2.0, 512, 0.75)
    bud = (24 * GB, 96 * GB)
    m.search_best_policy(sp, bud, 3)          # import/JIT warm
    m.search_best_policy(sp, bud, steps)      # full-size warm (allocator, page cache)
    best = None
    r = None
    for _ in range(3):
        t0 = time.process_time()
        r = m.search_best_policy(sp, bud, steps)
        dt = (time.process_time() - t0) * 1000.0
        best = dt if best is None else min(best, dt)
    print("%s %s" % (TOKEN, json.dumps({
        "timing_ms": round(best, 4), "steps": steps,
        "n_candidates": int(r["n_candidates"]), "n_feasible": int(r["n_feasible"]),
        "best_index": int(r["best_index"])})))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if cmd == "correctness":
        do_correctness()
    elif cmd == "timing":
        do_timing()
    else:
        raise SystemExit("usage: workload.py {correctness|timing}")
