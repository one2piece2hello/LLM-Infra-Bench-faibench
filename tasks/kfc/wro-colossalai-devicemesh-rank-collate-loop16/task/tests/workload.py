#!/usr/bin/env python3
"""Standalone CPU workload for the ColossalAI device-mesh rank-group collation
(colossalai.device.device_mesh.DeviceMesh._collate_global_ranks_in_same_process_group).

Given an N-dimensional logical device mesh, for a global rank this returns, per mesh axis, all
global ranks that share that axis line (the ranks in its process group along each axis). The
in-tree implementation resolves each coordinate back to its global rank by scanning ALL ranks in
the mesh, so collating every rank's groups (done at mesh construction) is quadratic in the number
of devices.

Two modes:
  correctness : build several meshes and check the per-axis rank groups for every global rank
                against an INDEPENDENT in-harness reference (row-major ravel/unravel of mesh
                coordinates). This reference is NOT in the editable scope.
  timing      : warmup + timed repeats of constructing a large logical mesh (construction runs the
                collation for every rank); paired candidate/baseline measurement by the verifier.

Emits one line `WRO_GDN_RESULT {json}`. Pure host logic (no GPU, no distributed init;
init_process_group defaults to False so construction issues no collective).
"""
import json
import sys
import time

import torch
from colossalai.device.device_mesh import DeviceMesh


def _ravel(coord, shape):
    idx = 0
    for c, s in zip(coord, shape):
        idx = idx * s + c
    return idx


def reference_groups(shape, global_rank):
    """Independent trusted reference (NOT in the editable scope): the mesh is row-major
    arange(prod(shape)).reshape(shape); a rank's coordinate is the unravel of its id; along axis
    `dim` its group is every rank agreeing on all other coordinates, ordered by that axis' local rank."""
    # unravel global_rank -> coordinate (row-major)
    base = []
    r = global_rank
    for s in reversed(shape):
        base.append(r % s); r //= s
    base = list(reversed(base))
    out = {}
    for dim in range(len(shape)):
        line = []
        for lr in range(shape[dim]):
            c = list(base); c[dim] = lr
            line.append(_ravel(c, shape))
        out[dim] = line
    return out


def make_mesh(shape):
    N = 1
    for s in shape:
        N *= s
    return DeviceMesh(torch.arange(0, N), mesh_shape=tuple(shape), device="cpu")


CORRECTNESS_SHAPES = [(4, 4), (2, 3, 4), (8, 8), (2, 2, 2, 2), (6, 5)]
TIMING_SHAPE = (8, 8, 2)   # 128 devices
WARMUP = 1
ITERS = 3


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"

    if mode == "correctness":
        ok = True
        details = []
        for shape in CORRECTNESS_SHAPES:
            mesh = make_mesh(shape)
            N = 1
            for s in shape:
                N *= s
            for g in range(N):
                got = mesh._collate_global_ranks_in_same_process_group(g)
                ref = reference_groups(shape, g)
                got_norm = {int(k): [int(x) for x in v] for k, v in got.items()}
                if got_norm != ref:
                    ok = False
                    details.append({"shape": list(shape), "rank": g, "got": got_norm, "ref": ref})
        res = {"mode": "correctness", "correctness_ok": bool(ok), "n_mismatch": len(details),
               "detail": details[:6]}
        print("WRO_GDN_RESULT " + json.dumps(res))
        sys.exit(0 if ok else 3)

    elif mode == "timing":
        for _ in range(WARMUP):
            make_mesh(TIMING_SHAPE)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            make_mesh(TIMING_SHAPE)
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print("WRO_GDN_RESULT " + json.dumps({
            "mode": "timing", "timing_ms": ms, "iters": ITERS, "shape": list(TIMING_SHAPE)}))
        sys.exit(0)
    else:
        print("WRO_GDN_RESULT " + json.dumps({"error": "bad_mode"}))
        sys.exit(2)


if __name__ == "__main__":
    main()
