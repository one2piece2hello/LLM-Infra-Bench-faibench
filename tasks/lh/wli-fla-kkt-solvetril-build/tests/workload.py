#!/usr/bin/env python3
"""Graded correctness driver for wli-fla-kkt-solvetril-build (cs321 build/hollow).

COUPLED MULTI-FILE scope = the chunk-local WY / UT transform in
fla-org/flash-linear-attention:
  fla/ops/common/chunk_scaled_dot_kkt.py :: chunk_scaled_dot_kkt_fwd   (STAGE-1 producer)
  fla/ops/utils/solve_tril.py            :: solve_tril                 (STAGE-2 consumer)

Data flow (as used verbatim by delta_rule/wy_fast.py, gated_delta_product/chunk.py,
gated_oja_rule/chunk.py):
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, chunk_size=BT)   # strict-lower(beta * k k^T)
    T = solve_tril(A)                                                  # (I + A)^{-1}
`T` is the transform matrix at the heart of delta-rule / gated-delta LINEAR-ATTENTION
chunking (an LLM sequence-mixing architecture). This grader loads the SOLVER's two
functions from /app/repo, composes them (producer -> consumer), and grades the transform
T against an INDEPENDENT pure-torch reference (build A, invert per chunk block) on a
weighted suite of randomized GPU cases.

Reward = sum(weight of passing cases) / sum(all weights) in [0,1] (dense gradient).
Reviewer-only; uploaded fresh at scoring; NEVER baked.

Per-file discriminative design (rule 12.5 / self-check ss3):
  * chunk_size == 16 cases: a single 16x16 diagonal sub-block -> solve_tril's off-diagonal
    "merge" step is empty, so these discriminate the PRODUCER (chunk_scaled_dot_kkt): they
    pass only once beta*k k^T is built correctly (incl. the 2**(g_i-g_j) gate on the gated
    cases). The natural partial (baseline2) passes these.
  * chunk_size in {32, 64} cases: the true inverse has non-zero OFF-diagonal 16x16
    sub-blocks -> these discriminate the CONSUMER (solve_tril): they stay RED until the
    inter-sub-block merge is implemented. baseline2 (diagonal-16-blocks-only) FAILS them.
Neither file alone captures the reward (superadditive): a hollow producer -> A garbage ->
every case fails; a hollow consumer -> raises -> every case fails.
"""
import json
import os
import sys

REPO = "/app/repo"
RATIO = 0.01  # relative RMS err tolerance; oracle (fp32 Triton) matches ref to ~1e-4 (repo CI)


def _emit(reward, gates, hard_fails, cases, detail=""):
    print("WRO_RESULT " + json.dumps({"score": round(float(reward), 6),
                                      "origin_ok": bool(gates.get("import_origin_ok", False))}))
    print(json.dumps({
        "mode": os.environ.get("KERNELBENCH_VERIFY_MODE", "noop"),
        "reward": round(float(reward), 6),
        "speedup": round(float(reward), 6),
        "correctness_ok": bool(reward >= 1.0 - 1e-6),
        "hard_fails": hard_fails,
        "gates": gates,
        "cases": cases,
        "detail": detail,
    }))
    sys.exit(0)


# --- weighted suite: 13 cases. max single w = 0.08; sum normalized. All T % chunk_size == 0.
#   GROUP A (chunk_size=16): producer-discriminative; baseline2 PASSES.
#   GROUP B (chunk_size in {32,64}): consumer-discriminative; baseline2 (diagonal-only) FAILS.
SUITE = [
    # --- GROUP A: single 16-sub-block (chunk_size=16) -> baseline2 passes ---
    dict(id="A_16_h2",    w=0.08, seed=1,  B=2, T=64,  H=2, D=64,  cs=16, gate=False),
    dict(id="A_16_d128",  w=0.08, seed=2,  B=2, T=128, H=2, D=128, cs=16, gate=False),
    dict(id="A_16_gated", w=0.08, seed=3,  B=2, T=64,  H=4, D=64,  cs=16, gate=True),
    dict(id="A_16_b1",    w=0.07, seed=4,  B=1, T=48,  H=1, D=64,  cs=16, gate=False),
    # --- GROUP B: chunk_size 32/64 -> off-diagonal merge needed; baseline2 FAILS ---
    dict(id="B_32_h2",    w=0.08, seed=5,  B=2, T=64,  H=2, D=64,  cs=32, gate=False),
    dict(id="B_32_gated", w=0.08, seed=6,  B=2, T=128, H=2, D=64,  cs=32, gate=True),
    dict(id="B_32_d128",  w=0.07, seed=7,  B=1, T=128, H=2, D=128, cs=32, gate=False),
    dict(id="B_32_b3",    w=0.06, seed=8,  B=3, T=96,  H=2, D=64,  cs=32, gate=False),
    dict(id="B_64_h2",    w=0.08, seed=9,  B=2, T=128, H=2, D=64,  cs=64, gate=False),
    dict(id="B_64_d128",  w=0.07, seed=10, B=1, T=256, H=1, D=128, cs=64, gate=False),
    dict(id="B_64_gated", w=0.08, seed=11, B=2, T=128, H=4, D=64,  cs=64, gate=True),
    dict(id="B_64_long",  w=0.07, seed=12, B=1, T=512, H=1, D=64,  cs=64, gate=False),
    dict(id="B_64_h4g",   w=0.06, seed=13, B=2, T=192, H=4, D=64,  cs=64, gate=True),
]


def build_case(torch, F, spec, device):
    g_ = torch.Generator(device=device).manual_seed(spec["seed"])
    B, T, H, D = spec["B"], spec["T"], spec["H"], spec["D"]
    # normalized k keeps I+A well-conditioned (mirrors tests/ops/test_solve_tril.py)
    k = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device=device, generator=g_), dim=-1, p=2)
    beta = torch.rand((B, T, H), dtype=torch.float32, device=device, generator=g_).sigmoid()
    if spec["gate"]:
        # g is a log2-space cumulative decay: monotone decreasing along T so 2**(g_i-g_j) in (0,1]
        # for i>j (the strictly-lower entries), keeping the transform well-conditioned.
        inc = torch.empty((B, T, H), dtype=torch.float32, device=device).uniform_(-0.25, -0.01, generator=g_)
        g = inc.cumsum(dim=1)
    else:
        g = None
    return k, beta, g, spec["cs"]


def ref_ut_transform(torch, k, beta, g, BT):
    """Independent reference: A[i,j] = beta_i * 2**(g_i-g_j) * (k_i . k_j) for i>j (strict lower),
    then T = (I + A)^{-1} per chunk block. Returns [B, T, H, BT]. Requires T % BT == 0."""
    B, T, H, D = k.shape
    assert T % BT == 0
    nc = T // BT
    kc = k.reshape(B, nc, BT, H, D).permute(0, 1, 3, 2, 4)          # [B,nc,H,BT,D]
    bc = beta.reshape(B, nc, BT, H).permute(0, 1, 3, 2)             # [B,nc,H,BT]
    A = torch.matmul(kc, kc.transpose(-1, -2))                     # [B,nc,H,BT,BT]  k_i . k_j
    if g is not None:
        gc = g.reshape(B, nc, BT, H).permute(0, 1, 3, 2)           # [B,nc,H,BT]
        A = A * torch.exp2(gc.unsqueeze(-1) - gc.unsqueeze(-2))    # 2**(g_i - g_j)
    A = A * bc.unsqueeze(-1)                                        # row scale by beta_i
    mask = torch.tril(torch.ones(BT, BT, device=k.device, dtype=torch.bool), -1)  # strict lower i>j
    A = A * mask
    eye = torch.eye(BT, device=k.device, dtype=torch.float32)
    T_ = torch.linalg.inv(eye + A)                                 # [B,nc,H,BT,BT]
    return T_.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)          # row i of block c -> token c*BT+i


def _origin_under_repo(mod, repo_real):
    """Resolve a module's file robustly: in this fla base `__file__` may be the RELATIVE
    path 'fla/...' (fla is found via the cwd entry), so anchor a relative path at /app/repo
    (cwd-independent). Returns (ok, resolved_path)."""
    f = getattr(mod, "__file__", "") or ""
    if not f:
        return False, "<no __file__>"
    cands = [f] if os.path.isabs(f) else [os.path.join(REPO, f), os.path.abspath(f)]
    for c in cands:
        rp = os.path.realpath(c)
        if rp.startswith(repo_real) and os.path.exists(rp):
            return True, rp
    return False, os.path.realpath(os.path.abspath(f))


def main():
    # Run from /app/repo so a RELATIVE fla.__file__ resolves under the repo (mirrors the
    # retention/kda/comba fla tasks, whose import-origin check runs with cwd=/app/repo).
    try:
        os.chdir(REPO)
    except Exception:
        pass
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # noqa
        _emit(0.0, {"scope_ok": True, "import_origin_ok": False, "correctness_ok": False}, ["torch_import:%r" % e], {})
    if not torch.cuda.is_available():
        _emit(0.0, {"scope_ok": True, "import_origin_ok": False, "correctness_ok": False}, ["no_cuda"], {})
    device = "cuda"
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ.setdefault("FLA_TRIL_PRECISION", "ieee")
    torch.manual_seed(42)

    # --- load the SOLVER's two scope functions from /app/repo ---
    # NOTE: fla/ops/utils/__init__.py does `from .solve_tril import solve_tril`, so the
    # ATTRIBUTE `fla.ops.utils.solve_tril` is the FUNCTION (shadowing the submodule). Import
    # the submodules and fetch the MODULE objects (and functions) from sys.modules so the
    # import-origin check sees a real module with __file__.
    try:
        import fla.ops.common.chunk_scaled_dot_kkt  # noqa
        import fla.ops.utils.solve_tril  # noqa
    except Exception as e:  # noqa
        _emit(0.0, {"scope_ok": True, "import_origin_ok": False, "correctness_ok": False},
              ["scope_import:%r" % e], {})
    m1 = sys.modules["fla.ops.common.chunk_scaled_dot_kkt"]
    m2 = sys.modules["fla.ops.utils.solve_tril"]
    chunk_scaled_dot_kkt_fwd = m1.chunk_scaled_dot_kkt_fwd
    solve_tril = m2.solve_tril

    # import-origin: both scope modules must resolve inside /app/repo (robust to relative __file__)
    repo_real = os.path.realpath(REPO)
    for mod, name in ((m1, "chunk_scaled_dot_kkt"), (m2, "solve_tril")):
        ok, resolved = _origin_under_repo(mod, repo_real)
        print("ORIGIN_DBG %s file=%r resolved=%s cwd=%s ok=%s" % (
            name, getattr(mod, "__file__", None), resolved, os.getcwd(), ok))
        if not ok:
            _emit(0.0, {"scope_ok": True, "import_origin_ok": False, "correctness_ok": False},
                  ["import_origin:%s=%s" % (name, resolved)], {})

    total = sum(c["w"] for c in SUITE)
    got = 0.0
    cases = {}
    for c in SUITE:
        try:
            k, beta, g, BT = build_case(torch, F, c, device)
            ref = ref_ut_transform(torch, k, beta, g, BT)
            A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=BT, output_dtype=torch.float32)
            tri = solve_tril(A)
            tri = tri.float()[:, :ref.shape[1]]
            err = (ref.detach().float() - tri.detach().float()).flatten().square().mean().sqrt().item()
            base = ref.detach().float().flatten().square().mean().sqrt().item()
            ratio = err / (base + 1e-8)
            nan = bool(torch.isnan(tri).any().item())
            ok = (not nan) and (ratio < RATIO)
            cases[c["id"]] = ("pass" if ok else "fail") + f"(ratio={ratio:.4f})"
            if ok:
                got += c["w"]
        except NotImplementedError:  # hollow / partial start
            cases[c["id"]] = "notimpl"
        except Exception as e:  # noqa
            cases[c["id"]] = "raised:%r" % e
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    reward = got / total
    gates = {"scope_ok": True, "import_origin_ok": True, "correctness_ok": True, "benchmark_ok": True}
    _emit(reward, gates, [], cases)


if __name__ == "__main__":
    main()
