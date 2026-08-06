"""Bytes-moved benchmark for gradient compression with feedback.

The value axis is *bytes moved*: the size of the compressed payload a worker must
transmit each step. Per shape:
    ratio = wire_bytes(baseline_payload) / wire_bytes(candidate_payload)
The final metric is the geometric mean of the per-shape ratios (prints
"speedup=X"). Byte counts are deterministic and device-independent, so this is a
stable, portable proxy — unlike wall time, which on a single process may not beat
a raw fp32 memcpy (the real bandwidth win is multi-node). The frozen baseline
(KB_BASELINE_MODULE) transmits the full fp32 buffer, so a no-op candidate (==
baseline) ties at ratio 1.0; a bit-packed sign + per-block scale payload moves
~1/32 the bytes.

Before measuring, the candidate is checked for the error-feedback identity on each
shape (new_residual == comp - decompress); a candidate that reports tiny bytes but
does not reconstruct a valid feedback estimate scores 0 here (and fails the
correctness suite).
"""

import importlib.util
import os
import sys

import torch

from kb_compress_harness import (
    FP32,
    dense_bytes,
    forbidden_vendor_guard,
    geomean,
    make_grad,
    wire_bytes,
)

# Gradient buffers from transformer weight shapes (flattened). numel 1e7-7e7.
SHAPES = [
    ("mlp_down", 4096, 11008, 7000),
    ("attn_qkv", 4096, 4096, 7100),
    ("wide", 8192, 8192, 7200),
    ("tall", 32768, 2048, 7300),
]

EF_TOL = 1e-2


def _load(path):
    spec = importlib.util.spec_from_file_location("kb_bench_mod_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_mod():
    path = os.environ.get("KB_BASELINE_MODULE", "/opt/verifier-baseline/grad_compress.py")
    return _load(path)


def _ef_ok(mod, buf, res):
    payload, new_res = mod.compress(buf, res)
    q = mod.decompress(payload)
    if tuple(q.shape) != tuple(buf.shape) or tuple(new_res.shape) != tuple(buf.shape):
        return False, payload
    comp = buf.to(FP32) + res.reshape(buf.shape).to(FP32)
    denom = comp.norm().item() or 1.0
    rel = (new_res.to(FP32) - (comp - q.to(FP32))).norm().item() / denom
    return (rel < EF_TOL), payload


def main():
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        sys.exit(2)
    torch.cuda.init()
    repo = os.environ.get("KB_REPO_DIR", "/app/repo")
    candidate = _load(os.path.join(repo, "grad_compress.py"))
    baseline = _baseline_mod()

    ratios = []
    for tag, R, C, seed in SHAPES:
        buf = make_grad((R, C), seed)
        res = torch.zeros_like(buf)
        numel = buf.numel()

        with forbidden_vendor_guard():
            ok, cand_payload = _ef_ok(candidate, buf, res)
        if not ok:
            print(f"BENCH_FAIL {tag}: candidate failed the feedback identity")
            print("speedup=0.0")
            sys.exit(1)
        _, base_payload = _ef_ok(baseline, buf, res)

        cand_bytes = wire_bytes(cand_payload)
        base_bytes = wire_bytes(base_payload)
        ratio = base_bytes / max(cand_bytes, 1)
        ratios.append(ratio)
        ideal = numel // 8  # 1 bit/element lower bound for the sign payload
        print(f"shape={tag} R={R} C={C} numel={numel} baseline_bytes={base_bytes} "
              f"candidate_bytes={cand_bytes} ratio={ratio:.4f} ideal_packed_bytes={ideal}")
        del buf, res, cand_payload, base_payload
        torch.cuda.empty_cache()

    print(f"speedup={geomean(ratios):.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
