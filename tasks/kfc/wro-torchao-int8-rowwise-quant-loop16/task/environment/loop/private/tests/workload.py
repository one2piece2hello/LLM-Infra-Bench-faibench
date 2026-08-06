#!/usr/bin/env python3
"""wro-torchao-int8-rowwise-quant — workload harness (reviewer-authored, uploaded with tests/).

Scope = torchao/prototype/quantized_training/int8.py (rowwise int8 quantization used in
int8 mixed-precision / quantized training).

Drives the PUBLIC entry `quantize_int8_rowwise` on a large 2D tensor on ONE GPU: each logical
row gets a symmetric absmax scale (|row|.amax()/127), then the row is quantized to int8.

Two modes:
  correctness : compare (int8 codes, scale) against an INDEPENDENT fp32 reference computed here
                (NOT part of the editable scope), by relative-norm tolerance on the dequant + scale.
  timing      : warmup + timed repeats of the quantization (paired vs the frozen baseline).

Emits one line `WRO_GDN_RESULT {json}`. The timed regime uses a large row count so the per-row
Python loop of the slow baseline separates clearly from a fused vectorized quantization.
"""
import json
import statistics
import sys

sys.path.insert(0, "/app/repo")
import torch

from torchao.prototype.quantized_training.int8 import quantize_int8_rowwise

DEV = "cuda"
ROWS = 4096
COLS = 4096
DTYPE = torch.bfloat16
REL_MAX_TOL = 2e-2
REL_L2_TOL = 1e-2
WARMUP = 3
ITERS = 10


def build_inputs(seed=0, rows=ROWS, cols=COLS, device=DEV):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, cols, device=device, dtype=DTYPE, generator=g)
    row_gain = torch.logspace(-2, 2, rows, device=device, dtype=torch.float32).view(-1, 1)
    return (x.float() * row_gain).to(DTYPE)


def run_scope(x):
    """Call the subsystem-under-test (candidate / degraded baseline code)."""
    return quantize_int8_rowwise(x, stochastic_rounding=False)


def int8_reference(x, eps=1e-12):
    """Independent trusted reference of rowwise int8 quantization (ground truth; NOT in the
    editable scope). Mirrors the documented numerics: scale=|x|.amax(1)/127, then
    round(x/scale) clipped to [-128,127] as int8. Output scale keeps the input dtype."""
    scale = x.abs().amax(1) / 127
    inv = 1.0 / scale.float().clip(eps)
    q = (x.float() * inv.view(-1, 1)).round().clip(-128, 127).to(torch.int8)
    return q, scale


def _relnorm(out, ref):
    diff = (out - ref).abs()
    max_abs = float(diff.max())
    denom = float(ref.abs().max().clamp_min(1e-6))
    return max_abs / denom, float(diff.norm() / ref.norm().clamp_min(1e-6))


def correctness():
    x = build_inputs(seed=0, rows=512, cols=1024)
    codes, scale = run_scope(x)
    ref_codes, ref_scale = int8_reference(x)
    torch.cuda.synchronize()
    # dequant parity (robust to +/-1 code rounding differences) + scale parity
    deq = codes.float() * scale.float().view(-1, 1)
    ref_deq = ref_codes.float() * ref_scale.float().view(-1, 1)
    rmo, rlo = _relnorm(deq, ref_deq)
    rms, rls = _relnorm(scale.float(), ref_scale.float())
    ncode_mismatch = int((codes.long() - ref_codes.long()).abs().gt(1).sum().item())
    ok = (rmo <= REL_MAX_TOL) and (rlo <= REL_L2_TOL) and (rms <= REL_MAX_TOL) and \
         (rls <= REL_L2_TOL) and (ncode_mismatch == 0)
    return {
        "correctness_ok": bool(ok), "rel_max_deq": rmo, "rel_l2_deq": rlo,
        "rel_max_scale": rms, "rel_l2_scale": rls, "n_code_mismatch_gt1": ncode_mismatch,
        "codes_shape": list(codes.shape), "scale_shape": list(scale.shape),
    }


def timing(iters=ITERS, warmup=WARMUP):
    x = build_inputs(seed=2)
    for _ in range(warmup):
        run_scope(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); run_scope(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if not torch.cuda.is_available():
        print("WRO_GDN_RESULT " + json.dumps({"error": "no_cuda"})); sys.exit(2)
    torch.cuda.synchronize()
    if mode == "correctness":
        res = correctness(); res["mode"] = "correctness"
        print("WRO_GDN_RESULT " + json.dumps(res)); sys.exit(0 if res["correctness_ok"] else 3)
    elif mode == "timing":
        ms = timing()
        print("WRO_GDN_RESULT " + json.dumps({"mode": "timing", "timing_ms": ms, "iters": ITERS,
              "rows": ROWS, "cols": COLS})); sys.exit(0)
    else:
        print("WRO_GDN_RESULT " + json.dumps({"error": "bad_mode"})); sys.exit(2)


if __name__ == "__main__":
    main()
