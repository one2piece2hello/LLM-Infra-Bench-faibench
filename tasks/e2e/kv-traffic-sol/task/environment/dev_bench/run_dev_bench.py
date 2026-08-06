#!/usr/bin/env python3
"""PUBLIC development bench — iterate on /app/repo/submission/kv_traffic.py.

Runs your engine over a PUBLIC workload set (different from the scored one), checks the
round-trip, and prints the measured step time and achieved bandwidth. Lower step time / higher
bandwidth is better. This file is yours; edit it freely — it is NOT the scored harness.

  python3 /app/dev_bench/run_dev_bench.py [--impl /app/repo/submission/kv_traffic.py]
                                          [--suite /app/dev_bench/dev_suite.json]
"""
import argparse
import importlib.util
import json
import sys

import torch

POISON = -12345.0


def load(path):
    spec = importlib.util.spec_from_file_location("dev_impl", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dev_impl"] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="/app/repo/submission/kv_traffic.py")
    ap.add_argument("--suite", default="/app/dev_bench/dev_suite.json")
    a = ap.parse_args()
    suite = json.load(open(a.suite))
    mod = load(a.impl)
    dev = "cuda"
    print("=" * 96)
    for cfg in suite["cases"]:
        B, L, P = cfg["batch"], cfg["num_layers"], cfg["page_size"]
        Hkv, D, NP = cfg["num_kv_heads"], cfg["head_size"], cfg["num_pages"]
        dt = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        seqs = list(cfg["seq_lens"])
        need = [(s + P - 1) // P for s in seqs]
        mp = max(need)
        order = torch.randperm(NP, generator=g)
        bt = torch.full((B, mp), -1, dtype=torch.int32)
        cur = 0
        for b in range(B):
            bt[b, :need[b]] = order[cur:cur + need[b]].to(torch.int32)
            cur += need[b]
        K = [[(torch.randn(seqs[b], Hkv, D, generator=g) * 0.5).to(dt).to(dev) for b in range(B)]
             for _ in range(L)]
        V = [[(torch.randn(seqs[b], Hkv, D, generator=g) * 0.5).to(dt).to(dev) for b in range(B)]
             for _ in range(L)]
        build = {"num_layers": L, "num_kv_heads": Hkv, "head_size": D, "page_size": P,
                 "num_pages": NP, "max_batch": B, "max_pages_per_request": mp,
                 "dtype": cfg.get("dtype", "bfloat16"), "device": dev}
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        eng = mod.KVTrafficEngine(build)
        eng.allocate()
        torch.cuda.synchronize()
        pool = torch.cuda.memory_allocated() - base

        ctx = torch.zeros(B, dtype=torch.int32)
        new = torch.tensor(seqs, dtype=torch.int32)
        T = int(new.sum())
        plan = {"block_table": bt.to(dev), "ctx_lens": ctx.to(dev), "new_lens": new.to(dev),
                "block_table_cpu": bt, "ctx_lens_cpu": ctx, "new_lens_cpu": new,
                "total_tokens": T, "batch": B}
        src = [(torch.cat([K[l][b] for b in range(B)]).contiguous(),
                torch.cat([V[l][b] for b in range(B)]).contiguous()) for l in range(L)]
        kout = [torch.empty(T, Hkv, D, dtype=dt, device=dev) for _ in range(L)]
        vout = [torch.empty_like(kout[0]) for _ in range(L)]

        def do_scatter():
            eng.begin_step(plan)
            for l in range(L):
                eng.scatter(l, src[l][0], src[l][1])

        def do_gather():
            eng.begin_step(plan)
            for l in range(L):
                eng.gather(l, kout[l], vout[l])

        ncp = int(cfg.get("n_copy_pages", 0))
        sp = order[:ncp].to(torch.int32).to(dev) if ncp else None
        dpg = order[NP - ncp:].to(torch.int32).to(dev) if ncp else None

        def do_copy():
            for l in range(L):
                eng.copy_pages(l, sp, dpg)

        do_scatter()
        for l in range(L):
            kout[l].fill_(POISON)
            vout[l].fill_(POISON)
        do_gather()
        torch.cuda.synchronize()
        ok = all(torch.equal(kout[l], src[l][0]) and torch.equal(vout[l], src[l][1])
                 for l in range(L))

        def timed(fn, reps=8):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                e0 = torch.cuda.Event(True)
                e1 = torch.cuda.Event(True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / 1e3)
            ts.sort()
            return ts[len(ts) // 2]

        elt = 2
        by = 2 * 2 * L * T * Hkv * D * elt
        tg = timed(do_gather)
        ts_ = timed(do_scatter)
        line = ("%-18s B=%-2d L=%d page=%-3d row=%4dB | gather %8.3f ms %8.1f GB/s | "
                "scatter %8.3f ms %8.1f GB/s" % (cfg["case_id"], B, L, P, Hkv * D * elt,
                                                 tg * 1e3, by / tg / 1e9, ts_ * 1e3,
                                                 by / ts_ / 1e9))
        if ncp:
            bc = 2 * 2 * L * ncp * P * Hkv * D * elt
            tc = timed(do_copy)
            line += " | copy_pages %8.3f ms %8.1f GB/s" % (tc * 1e3, bc / tc / 1e9)
        print(line)
        print("%-18s pool=%.2f GB   round-trip bit-exact: %s" % ("", pool / 1e9, ok))
        del eng, K, V, src, kout, vout
        torch.cuda.empty_cache()
    print("=" * 96)
    print("higher GB/s = better; the scored workloads differ from these")


if __name__ == "__main__":
    main()
