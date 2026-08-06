#!/usr/bin/env python3
"""H20 Speed-of-Light (SOL) helper for `sol_fraction` perf metrics.

WHAT THIS IS
------------
A single-file, stdlib-only helper used to (1) anchor a
performance task to the hardware Speed-of-Light and (2) report
`sol_fraction = achieved / SOL` so stronger optimizations keep scoring higher
toward the theoretical ceiling (maximal score discrimination).

The Roofline SOL for a kernel is:

    T_SOL(seconds) = max( FLOPs / peak_dense_FLOPs ,  bytes_moved / peak_HBM_BW )
    sol_fraction   = T_SOL / T_achieved            # in (0, 1]; 1.0 == hit the roofline

TWO PEAK SOURCES (use the empirical one when present)
-----------------------------------------------------
1. SPEC-SHEET peaks (`H20_SPEC` below): vendor dense numbers. Best-effort, and
   REAL kernels never reach them (a good GEMM lands ~0.75-0.9x dense peak; a big
   HBM copy lands ~0.85-0.95x spec BW). Use as a fallback / sanity ceiling.
2. EMPIRICAL peaks MEASURED ON A REAL H20 by `measure_peaks()` (a cublas
   GEMM sweep + a large device-copy). This is what we mean by
   "MEASURE the H20 theoretical peak" — the achievable peak, cached to
   `h20_measured_peaks.json`. `load_peaks()` prefers it automatically.

🔴 EXECUTION DISCIPLINE. The measurement path imports torch and launches GPU
work — it needs a real GPU. `measure_peaks()` / `python3 h20.py measure` MUST run
inside a container with an H20 attached, NEVER on a CPU-only front-end host
(it will OOM there). The pure-math paths (`roofline_t_sol`,
`sol_fraction`, `H20_SPEC`, `load_peaks`) import nothing heavy and are safe
anywhere.

USAGE
-----
    # in a task verifier (pure math — safe anywhere):
    from h20 import roofline_t_sol, sol_fraction, load_peaks
    peaks = load_peaks()                       # empirical if cached, else spec
    t_sol = roofline_t_sol(flops=F, bytes_moved=B, dtype="bf16", peaks=peaks)
    frac  = sol_fraction(t_achieved_s=measured_s, flops=F, bytes_moved=B,
                         dtype="bf16", peaks=peaks)

    # ON AN H20 ONLY — measure achievable peaks once per base, cache them:
    #   python3 h20.py measure --out /app/tests/h20_measured_peaks.json
    # compute a SOL from the CLI:
    #   python3 h20.py sol --flops 4.4e12 --bytes 3.2e9 --dtype bf16 --achieved-ms 1.9
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# H20 SPEC-SHEET PEAKS (NVIDIA H20, Hopper GH100 die, memory-full/compute-cut).
# 🔴 THESE ARE VENDOR DENSE NUMBERS — a sanity ceiling, NOT the achievable peak.
# Prefer empirically measured peaks (measure_peaks()); real kernels land well
# below these. If your node reports different numbers, TRUST the measured
# values and (optionally) correct this table with a source.
#   TFLOPS = 1e12 FLOP/s ; HBM bandwidth in GB/s = 1e9 byte/s.
# Sources: NVIDIA H20 product brief / commonly reported figures (verify on your card).
# ---------------------------------------------------------------------------
H20_SPEC = {
    "hbm_gb": 96.0,
    "hbm_gbps": 4000.0,          # ~4.0 TB/s HBM3 (SPEC; measured copy ~0.85-0.95x)
    "peak_tflops": {             # DENSE tensor-core / vector peaks (SPEC)
        "fp8": 296.0,            # E4M3/E5M2 tensor core
        "int8": 296.0,           # TOPS
        "fp16": 148.0,           # FP16 tensor core
        "bf16": 148.0,           # BF16 tensor core
        "tf32": 74.0,            # TF32 tensor core
        "fp32": 44.0,            # FP32 (non-tensor / FMA vector)
        "fp64": 1.0,             # FP64 (heavily cut on H20)
    },
    # Rough achievable-efficiency hints (measured/spec) for sanity checks only:
    "achievable_hint": {"gemm": 0.80, "hbm_copy": 0.90},
    "_note": "SPEC-SHEET dense peaks; verify with measure_peaks() on your H20.",
    "_source": "vendor brief (best-effort); empirical override preferred",
}

DTYPE_ALIASES = {
    "bfloat16": "bf16", "float16": "fp16", "half": "fp16",
    "float32": "fp32", "float": "fp32", "double": "fp64", "float64": "fp64",
    "e4m3": "fp8", "e5m2": "fp8", "float8": "fp8",
}

_CACHE_BASENAME = "h20_measured_peaks.json"


def _norm_dtype(dtype: str) -> str:
    d = str(dtype).lower().strip()
    return DTYPE_ALIASES.get(d, d)


def load_peaks(measured_path: Optional[str] = None) -> dict:
    """Return the peak table to use: empirical (if cached & readable) else spec.

    Search order for the empirical cache: `measured_path` arg → env
    H20_MEASURED_PEAKS → ./h20_measured_peaks.json → alongside this file.
    The returned dict always has keys {hbm_gbps, peak_tflops{...}, _origin}.
    Pure I/O of a small JSON — safe anywhere (imports nothing heavy).
    """
    candidates = []
    if measured_path:
        candidates.append(measured_path)
    if os.environ.get("H20_MEASURED_PEAKS"):
        candidates.append(os.environ["H20_MEASURED_PEAKS"])
    candidates.append(os.path.join(os.getcwd(), _CACHE_BASENAME))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), _CACHE_BASENAME))
    for p in candidates:
        try:
            with open(p) as f:
                m = json.load(f)
            if "peak_tflops" in m and "hbm_gbps" in m:
                m.setdefault("_origin", f"measured:{p}")
                return m
        except (OSError, ValueError):
            continue
    spec = {
        "hbm_gb": H20_SPEC["hbm_gb"],
        "hbm_gbps": H20_SPEC["hbm_gbps"],
        "peak_tflops": dict(H20_SPEC["peak_tflops"]),
        "_origin": "spec_sheet",
    }
    return spec


def peak_flops(dtype: str, peaks: Optional[dict] = None) -> float:
    """Peak FLOP/s for a dtype (from measured peaks if given, else spec)."""
    peaks = peaks or load_peaks()
    d = _norm_dtype(dtype)
    tf = peaks["peak_tflops"]
    if d not in tf:
        raise KeyError(f"unknown dtype {dtype!r}; known: {sorted(tf)}")
    return tf[d] * 1e12


def peak_bw(peaks: Optional[dict] = None) -> float:
    """Peak HBM bandwidth in byte/s (from measured peaks if given, else spec)."""
    peaks = peaks or load_peaks()
    return peaks["hbm_gbps"] * 1e9


def roofline_t_sol(flops: float, bytes_moved: float, dtype: str = "bf16",
                   peaks: Optional[dict] = None) -> dict:
    """Roofline Speed-of-Light time (seconds) + which term binds.

    T_SOL = max(FLOPs/peak_flops, bytes_moved/peak_bw).
    Returns {t_sol_s, t_compute_s, t_memory_s, bound, arithmetic_intensity,
             ridge_point_flops_per_byte, peaks_origin}. Pure math — safe anywhere.
    """
    pf = peak_flops(dtype, peaks)
    pb = peak_bw(peaks)
    t_compute = (flops / pf) if flops > 0 else 0.0
    t_memory = (bytes_moved / pb) if bytes_moved > 0 else 0.0
    t_sol = max(t_compute, t_memory)
    ai = (flops / bytes_moved) if bytes_moved > 0 else float("inf")
    ridge = pf / pb  # FLOP/byte where the roofline turns from BW- to compute-bound
    return {
        "t_sol_s": t_sol,
        "t_compute_s": t_compute,
        "t_memory_s": t_memory,
        "bound": "compute" if t_compute >= t_memory else "memory",
        "arithmetic_intensity_flops_per_byte": ai,
        "ridge_point_flops_per_byte": ridge,
        "peaks_origin": (peaks or load_peaks()).get("_origin", "?"),
    }


def sol_fraction(t_achieved_s: float, flops: float, bytes_moved: float,
                 dtype: str = "bf16", peaks: Optional[dict] = None) -> float:
    """achieved / SOL = T_SOL / T_achieved, clamped to (0, 1] when achieved>0.

    1.0 means the kernel hit the Roofline; 0.25 means 4x off the ceiling. This
    is the value reported as the perf metric (still oracle-gated at
    1.0 downstream). Pure math — safe anywhere.
    """
    if t_achieved_s <= 0:
        raise ValueError("t_achieved_s must be > 0")
    r = roofline_t_sol(flops, bytes_moved, dtype, peaks)
    return r["t_sol_s"] / t_achieved_s


# ---------------------------------------------------------------------------
# EMPIRICAL peak measurement — 🔴 GPU-ONLY (imports torch, launches GPU work).
# ---------------------------------------------------------------------------
def measure_peaks(dtypes=("bf16", "fp16"), sizes=(4096, 8192, 16384),
                  iters: int = 50, copy_gb: float = 2.0) -> dict:
    """Microbench achievable GEMM TFLOPS + HBM copy GB/s on the current GPU.

    🔴 GPU-ONLY: run this inside a container with an H20 attached, never on a CPU-only host.
    Imports torch here (not at module top) so the pure-math paths stay importable
    on a torch-less host. Returns a peaks dict compatible with load_peaks()/
    roofline_t_sol(), tagged `_origin: "measured:<gpu>"`.
    """
    import torch  # noqa: local import — heavy, GPU-path only

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — measure_peaks() must run on an H20")
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    if "H20" not in name.upper():
        # Not fatal (some nodes label the device differently) but record it loudly.
        sys.stderr.write(f"[h20.py] WARNING: device is {name!r}, expected H20\n")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}

    def _time(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters / 1e3  # seconds/iter

    peak_tflops = {}
    for dt in dtypes:
        if dt not in torch_dtype:
            continue
        best = 0.0
        for n in sizes:
            a = torch.randn(n, n, device=dev, dtype=torch_dtype[dt])
            b = torch.randn(n, n, device=dev, dtype=torch_dtype[dt])
            try:
                s = _time(lambda: torch.matmul(a, b))
            except RuntimeError:
                continue
            flop = 2.0 * n * n * n  # multiply-add
            best = max(best, flop / s / 1e12)
            del a, b
            torch.cuda.empty_cache()
        if best > 0:
            peak_tflops[dt] = round(best, 2)

    # HBM bandwidth via a large device-to-device copy (read+write => 2x bytes).
    n_elems = int(copy_gb * 1e9 / 4)  # fp32 elements
    src = torch.empty(n_elems, device=dev, dtype=torch.float32)
    dst = torch.empty_like(src)
    s = _time(lambda: dst.copy_(src))
    hbm_gbps = round((2.0 * n_elems * 4) / s / 1e9, 1)
    del src, dst
    torch.cuda.empty_cache()

    # keep any spec dtypes we did not measure so the table stays complete
    merged = dict(H20_SPEC["peak_tflops"])
    merged.update(peak_tflops)
    return {
        "hbm_gb": H20_SPEC["hbm_gb"],
        "hbm_gbps": hbm_gbps,
        "peak_tflops": merged,
        "_origin": f"measured:{name}",
        "_measured_dtypes": list(peak_tflops.keys()),
        "_iters": iters,
        "_sizes": list(sizes),
    }


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="H20 SOL helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="[H20 GPU ONLY] microbench achievable peaks")
    m.add_argument("--out", default=_CACHE_BASENAME, help="write peaks JSON here")
    m.add_argument("--iters", type=int, default=50)

    s = sub.add_parser("sol", help="compute T_SOL / sol_fraction (pure math)")
    s.add_argument("--flops", type=float, required=True)
    s.add_argument("--bytes", type=float, required=True, dest="bytes_moved")
    s.add_argument("--dtype", default="bf16")
    s.add_argument("--achieved-ms", type=float, default=None,
                   help="if given, also print sol_fraction")
    s.add_argument("--peaks", default=None, help="path to measured peaks JSON")

    p = sub.add_parser("peaks", help="print the peak table in effect")
    p.add_argument("--peaks", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "measure":
        peaks = measure_peaks(iters=args.iters)
        with open(args.out, "w") as f:
            json.dump(peaks, f, indent=2)
        print(json.dumps(peaks, indent=2))
        print(f"\n[h20.py] wrote {args.out}", file=sys.stderr)
    elif args.cmd == "peaks":
        print(json.dumps(load_peaks(args.peaks), indent=2))
    elif args.cmd == "sol":
        peaks = load_peaks(args.peaks)
        r = roofline_t_sol(args.flops, args.bytes_moved, args.dtype, peaks)
        out = dict(r)
        out["t_sol_ms"] = r["t_sol_s"] * 1e3
        if args.achieved_ms is not None:
            frac = sol_fraction(args.achieved_ms / 1e3, args.flops,
                                args.bytes_moved, args.dtype, peaks)
            out["achieved_ms"] = args.achieved_ms
            out["sol_fraction"] = frac
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
