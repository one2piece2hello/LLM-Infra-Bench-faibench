#!/usr/bin/env python3
"""PUBLIC development bench (model-visible) — run your implementation over a small PUBLIC
workload set, check it against a float32 causal reference, and print the per-call time and the
achieved tensor-core throughput.

    python3 /app/dev_bench/run_dev_bench.py [--impl PATH]

The scored workloads are NOT these. A lower per-call time here usually means a higher score, but
the scored set has different shapes.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=os.environ.get("SUBMISSION_DIR", "/app/submission")
                    + "/varlen_prefill_attn.py")
    ap.add_argument("--suite", default=os.path.join(HERE, "dev_suite.json"))
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("dev_impl", a.impl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "VarlenPrefillAttention")

    suite = json.load(open(a.suite))
    print("device:", torch.cuda.get_device_name(0))

    def build(cfg):
        sl = [int(x) for x in cfg["seq_lens"]]
        Hq, Hkv, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_size"]
        tot, mx = sum(sl), max(sl + [0])
        g = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
        mk = lambda h: (torch.randn(max(tot, 1), h, D, generator=g) * 0.5).to(
            torch.bfloat16).cuda()[:tot].contiguous()
        q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
        cu = torch.zeros(len(sl) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(sl, dtype=torch.int32), 0)
        impl = cls({"num_q_heads": Hq, "num_kv_heads": Hkv, "head_size": D,
                    "dtype": "bfloat16", "device": "cuda", "max_num_seqs": len(sl),
                    "max_seq_len": max(mx, 1), "max_total_tokens": max(tot, 1), "causal": True,
                    "softmax_scale": 1.0 / math.sqrt(D)})
        impl.prepare()
        return impl, q, k, v, cu.cuda(), sl, Hq, Hkv, D, tot, mx

    def ref_row(q, k, v, aa, r, Hq, Hkv, D):
        rep = Hq // Hkv
        kk = k[aa:aa + r + 1].float(); vv = v[aa:aa + r + 1].float()
        if rep > 1:
            kk = kk.repeat_interleave(rep, dim=1); vv = vv.repeat_interleave(rep, dim=1)
        lo = torch.einsum("hd,shd->hs", q[aa + r].float(), kk) / math.sqrt(D)
        return torch.einsum("hs,shd->hd", torch.softmax(lo, -1), vv)

    print("\n--- public degenerate-shape checks (the scored run has more of these) ---")
    for cfg in suite.get("edges", []):
        impl, q, k, v, cu, sl, Hq, Hkv, D, tot, mx = build(cfg)
        out = torch.empty(tot, Hq, D, device="cuda", dtype=torch.bfloat16)
        try:
            ret = impl.forward(q, k, v, cu, mx, out)
            cul = [0]
            for s_ in sl:
                cul.append(cul[-1] + s_)
            worst = 0.0
            for i, s_ in enumerate(sl):
                for r in range(s_):
                    ref = ref_row(q, k, v, cul[i], r, Hq, Hkv, D)
                    rms = ref.pow(2).mean().sqrt().item()
                    worst = max(worst, (ret[cul[i] + r].float() - ref).abs().max().item()
                                / (rms + 1e-3))
            print("  %-22s seq_lens=%-26s worst deviation %.4f" % (cfg["case_id"], str(sl), worst))
        except Exception as e:
            print("  %-22s seq_lens=%-26s RAISED %s: %s" % (cfg["case_id"], str(sl),
                                                            type(e).__name__, str(e)[:120]))
        del impl
        torch.cuda.empty_cache()

    rows = []
    for cfg in suite["cases"]:
        impl, q, k, v, cu, sl, Hq, Hkv, D, tot, mx = build(cfg)
        dt = torch.bfloat16
        out = torch.empty(tot, Hq, D, device="cuda", dtype=dt)
        ret = impl.forward(q, k, v, cu, mx, out)

        # float32 causal reference on a few rows of every sequence
        cul = [0]
        for s in sl:
            cul.append(cul[-1] + s)
        worst = 0.0
        for i, s in enumerate(sl):
            for r in (0, s // 2, s - 1):
                ref = ref_row(q, k, v, cul[i], r, Hq, Hkv, D)
                rms = ref.pow(2).mean().sqrt().item()
                worst = max(worst, (ret[cul[i] + r].float() - ref).abs().max().item()
                            / (rms + 1e-3))

        for _ in range(3):
            impl.forward(q, k, v, cu, mx, out)
        torch.cuda.synchronize()
        ts = []
        for _ in range(6):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(4):
                impl.forward(q, k, v, cu, mx, out)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3 / 4)
        ts.sort()
        t = ts[len(ts) // 2]
        flops = sum(4.0 * Hq * D * (s * (s + 1) / 2.0) for s in sl)
        rows.append((cfg["case_id"], t, flops / t / 1e12, worst))
        del impl, q, k, v, out
        torch.cuda.empty_cache()

    print("\n%-22s %12s %14s %12s" % ("case", "ms/call", "TFLOP/s", "parity"))
    for (cid, t, tf, w) in rows:
        print("%-22s %12.3f %14.2f %12.4f%s" % (cid, t * 1e3, tf, w,
                                                "  <-- WRONG OUTPUT" if w > 1.0 else ""))
    print("\n(parity = the largest elementwise deviation from the float32 causal reference, "
          "divided by the RMS of that row's reference. An implementation that computes the exact "
          "causal attention in bfloat16 lands FAR below the scored tolerance; a value of order 1 "
          "means the output is simply wrong. The scored run uses the same measure, over hidden "
          "rows plus degenerate shapes.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
