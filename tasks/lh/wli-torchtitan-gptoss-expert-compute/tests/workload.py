#!/usr/bin/env python3
"""wli-torchtitan-gptoss-expert-compute -- GRADED correctness harness (reviewer-authored).

Scope (COUPLED, multi-file, DEGRADED start-shape):
  * torchtitan/models/common/moe.py      (PRODUCER: MoE.forward builds the one-hot routing
                                           map + per-expert token counts + the running
                                           tokens_per_expert load statistic)
  * torchtitan/models/gpt_oss/moe.py      (CONSUMER: swiglu activation + GptOssGroupedExperts.
                                           _experts_forward grouped-mm expert compute that
                                           CONSUMES the per-expert token counts)

The shipped tree is FUNCTIONALLY DEGRADED at 6 coupled points (each still RUNS; hidden cases
fail).  This harness runs the REAL torch MoE producer + gpt-oss expert compute on small CPU
tensors and compares against an in-harness reference implementation.  ``torch._grouped_mm``
(a hardware grouped-matmul primitive with no CPU kernel) is neutralized to a faithful CPU
segment-matmul so the SURROUNDING scope logic -- offsets, bias placement, swiglu, routing
counts -- is what is graded; BOTH the scope and the reference call the same neutralized
primitive, so the only differences come from the degraded scope logic.

Points (each >=1 discriminative hidden case; per-file discriminative cases present):
  moe.py (PRODUCER -- MoE.forward routing bookkeeping)
    P1 routing_map one-hot           : scatter ALL top-k expert ids (not just the first) into
                                        the (B,L,E) one-hot map, so per-expert counts include
                                        every routed expert.
    P2 tokens_per_expert accumulate  : the running load counter ACCUMULATES (add_) across
                                        forward calls, not overwrites (copy_).
  gpt_oss/moe.py (CONSUMER -- swiglu + grouped expert compute)
    P3 swiglu gate/linear split      : gate/linear are the INTERLEAVED even/odd channels
                                        (x[...,::2] / x[...,1::2]), not contiguous halves.
    P4 swiglu linear bias-of-1       : the linear branch carries an extra +1 bias
                                        (out_glu * (1 + x_linear)  ==  addcmul).
    P5 swiglu glu clamp              : the glu branch is clamped ONE-SIDED (max only), the
                                        linear branch two-sided.
    P6 mlp2 output bias              : the down-projection (mlp2) output bias is added back.

COUPLING (moe.py <-> gpt_oss/moe.py): GptOssGroupedExperts._experts_forward turns the
per-expert token counts MoE.forward produces into the grouped-mm segment offsets and the
repeat-interleaved per-token bias; a correct expert output therefore requires BOTH the
producer to count experts correctly (P1) AND the consumer to split/clamp/bias/compute
correctly (P3-P6).  The end-to-end cases run scope MoE.forward -> scope _experts_forward and
compare to a fully-correct reference: they stay red unless BOTH files are correct.  Restoring
only the producer (moe.py) still leaves the gpt-oss compute wrong; neither file is verifiable
alone.

Emits ONE ``WRO_RESULT {json}`` line: graded ``score`` in [0,1], both modules' resolved
``__file__`` (import-origin), a per-case trace.  Single process, CPU-only, no GPU, no weights.
"""
import sys
import json
import types
import contextlib
import importlib.util
from unittest.mock import MagicMock

REPO = "/app/repo"
REL_MOE = "torchtitan/models/common/moe.py"
REL_GO = "torchtitan/models/gpt_oss/moe.py"

import torch  # REAL torch — the producer/consumer run genuine tensor ops on CPU.

# CPU float32 autocast is unsupported; precision is irrelevant here (fixed inputs), so make
# autocast a no-op context (mirrors the sibling moe-routing harness).
torch.autocast = lambda *a, **k: contextlib.nullcontext()


def _ref_grouped_mm(a, b, offs=None):
    """Faithful CPU stand-in for ``torch._grouped_mm(a, b, offs=cumulative_ends)``.

    a: (R, Kin); b: (E, Kin, Kout); offs: (E,) cumulative END row indices into a.
    Expert e multiplies rows [start:offs[e]] of a by b[e]. Robust to non-monotone /
    out-of-range offs (clamped), so a degraded-offset candidate runs instead of crashing.
    Computes in float32 then casts back to a.dtype -- identical for scope and reference.
    """
    R = a.shape[0]
    Kout = b.shape[-1]
    out = a.new_zeros((R, Kout))
    if offs is None:
        return out
    start = 0
    for e, end in enumerate([int(v) for v in offs.tolist()]):
        s = max(0, min(int(start), R))
        en = max(s, min(int(end), R))
        if en > s and e < b.shape[0]:
            out[s:en] = (a[s:en].float() @ b[e].float()).to(a.dtype)
        start = end
    return out


torch._grouped_mm = _ref_grouped_mm


def _register_stubs():
    def pkg(name, path=None):
        m = types.ModuleType(name)
        if path:
            m.__path__ = [path]
        sys.modules[name] = m
        return m

    # bare `import spmd_types as spmd` marker module (used only outside tested paths).
    spmd = types.ModuleType("spmd_types")
    spmd.R = object(); spmd.P = object(); spmd.V = object(); spmd.I = object()
    spmd.S = lambda d: ("S", d)
    spmd.Shard = tuple
    spmd.is_type_checking = lambda: False
    spmd.mutate_type = lambda *a, **k: None
    spmd.assert_type = lambda *a, **k: None
    spmd.local = lambda *a, **k: contextlib.nullcontext()
    spmd.register_autograd_function = lambda cls: cls  # identity decorator for ScaleBiasForward
    sys.modules["spmd_types"] = spmd

    pkg("torchtitan", REPO + "/torchtitan")
    pkg("torchtitan.distributed", REPO + "/torchtitan/distributed")

    tsp = types.ModuleType("torchtitan.distributed.spmd_types")
    tsp.maybe_set_sparse_mesh = lambda *a, **k: contextlib.nullcontext()
    tsp.spmd_mesh_size = lambda *a, **k: 1
    tsp.current_spmd_mesh = lambda *a, **k: None
    sys.modules["torchtitan.distributed.spmd_types"] = tsp

    tu = types.ModuleType("torchtitan.distributed.utils")
    tu.get_spmd_backend = lambda *a, **k: "default"
    sys.modules["torchtitan.distributed.utils"] = tu

    # config.Configurable (base of dispatchers, not exercised) + Module base (bypass __init__).
    cfg = types.ModuleType("torchtitan.config")
    class _Configurable:
        class Config:
            def build(self, **kw):
                return None
    cfg.Configurable = _Configurable
    sys.modules["torchtitan.config"] = cfg

    pkg("torchtitan.ops", REPO + "/torchtitan/ops")
    sa = types.ModuleType("torchtitan.ops.scatter_add")
    sa.deterministic_scatter_add = lambda out, index, src: out.scatter_add(0, index, src)
    sys.modules["torchtitan.ops.scatter_add"] = sa

    pkg("torchtitan.tools", REPO + "/torchtitan/tools")
    ttu = types.ModuleType("torchtitan.tools.utils")
    ttu.device_type = "cpu"; ttu.device_module = MagicMock()
    sys.modules["torchtitan.tools.utils"] = ttu

    pkg("torchtitan.protocols")
    pm = types.ModuleType("torchtitan.protocols.module")
    class _Module:
        class Config:
            def build(self, **kw):
                return None
    pm.Module = _Module
    sys.modules["torchtitan.protocols.module"] = pm

    models = pkg("torchtitan.models", REPO + "/torchtitan/models")
    common = pkg("torchtitan.models.common", REPO + "/torchtitan/models/common")
    models.common = common
    ff = types.ModuleType("torchtitan.models.common.feed_forward")
    ff.FeedForward = type("FeedForward", (), {"Config": type("Config", (), {})})
    ff.compute_ffn_hidden_dim = lambda *a, **k: 0
    sys.modules["torchtitan.models.common.feed_forward"] = ff
    nnm = types.ModuleType("torchtitan.models.common.nn_modules")
    nnm.Linear = type("Linear", (), {"Config": type("Config", (), {})})
    nnm.RMSNorm = type("RMSNorm", (), {"Config": type("Config", (), {})})
    sys.modules["torchtitan.models.common.nn_modules"] = nnm

    # token_dispatcher is OUT OF SCOPE for this task -- stub the classes MoE.forward
    # references.  common/moe.py has NO `from __future__ import annotations`, so its dataclass
    # annotation `token_dispatcher: LocalTokenDispatcher.Config` is EVALUATED at class-def time:
    # the stub classes must expose a `.Config` attribute or moe.py fails to import.
    td = types.ModuleType("torchtitan.models.common.token_dispatcher")
    _cfg = type("Config", (), {})
    td.LocalTokenDispatcher = type("LocalTokenDispatcher", (), {"Config": _cfg})
    td.DeepEPTokenDispatcher = type("DeepEPTokenDispatcher", (), {"Config": _cfg})
    td.AllToAllTokenDispatcher = type("AllToAllTokenDispatcher", (), {"Config": _cfg})
    sys.modules["torchtitan.models.common.token_dispatcher"] = td

    gptoss = pkg("torchtitan.models.gpt_oss", REPO + "/torchtitan/models/gpt_oss")
    models.gpt_oss = gptoss
    return common, gptoss


def _load_from(rel, modname):
    spec = importlib.util.spec_from_file_location(modname, REPO + "/" + rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def load_modules():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    common, gptoss = _register_stubs()
    moe = _load_from(REL_MOE, "torchtitan.models.common.moe")
    common.moe = moe
    go = _load_from(REL_GO, "torchtitan.models.gpt_oss.moe")
    gptoss.moe = go
    return moe, go


# --------------------------------------------------------------------------------------
# Reference implementations (CORRECT gpt-oss expert compute + MoE routing bookkeeping).
# --------------------------------------------------------------------------------------
def ref_swiglu(x, alpha=1.702, limit=7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return torch.addcmul(out_glu, out_glu, x_linear)


def ref_experts_forward(mlp1_w, mlp1_b, mlp2_w, mlp2_b, x_RD, counts, limit):
    R = x_RD.shape[0]
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    tail = (R - offsets[-1]).unsqueeze(0).to(counts.dtype)
    counts_long = torch.cat([counts, tail]).long()
    h_RG = _ref_grouped_mm(x_RD.bfloat16(), mlp1_w.transpose(-2, -1).bfloat16(), offs=offsets)
    b1 = torch.cat([mlp1_b, mlp1_b.new_zeros(1, mlp1_b.shape[-1])])
    b1_RG = b1.repeat_interleave(counts_long, dim=0, output_size=R)
    h_RG = h_RG + b1_RG.to(h_RG.dtype)
    h_RF = ref_swiglu(h_RG, limit=limit)
    h_RD = _ref_grouped_mm(h_RF, mlp2_w.transpose(-2, -1).bfloat16(), offs=offsets)
    b2 = torch.cat([mlp2_b, mlp2_b.new_zeros(1, mlp2_b.shape[-1])])
    b2_RD = b2.repeat_interleave(counts_long, dim=0, output_size=R)
    h_RD = h_RD + b2_RD.to(h_RD.dtype)
    return h_RD


def ref_counts(topk_ids_BLK, num_experts):
    B, L, K = topk_ids_BLK.shape
    scores = torch.zeros(B, L, num_experts)
    rmap = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_ids_BLK, True)
    return rmap.sum(dim=(0, 1))


# --------------------------------------------------------------------------------------
# Instantiate scope classes WITHOUT their real __init__ (bypass Config/nn.Module machinery).
# --------------------------------------------------------------------------------------
class _FakeExperts:
    """Stands in for MoE.experts: captures the per-expert counts MoE.forward passes in."""
    def __init__(self, B, L, D):
        self.B, self.L, self.D = B, L, D
        self.token_dispatcher = object()  # not a DeepEPTokenDispatcher; sp_size default 1
        self.captured = None

    def __call__(self, x_BLD, topk_scores_BLK, topk_expert_ids_BLK,
                 num_local_tokens_per_expert_E, *, num_local_tokens_after_seq_dim_padding):
        self.captured = num_local_tokens_per_expert_E.clone()
        return torch.zeros(self.B, self.L, self.D)


def make_moe(moe, *, num_experts, load_balance_coeff=1e-3):
    M = moe.MoE.__new__(moe.MoE)
    M.seq_dim_tp_sharded = False
    M.shared_experts = None
    M.expert_bias_E = None
    M.load_balance_coeff = load_balance_coeff
    M.tokens_per_expert_E = torch.zeros(num_experts, dtype=torch.float32)
    return M


class _FixedRouter:
    def __init__(self, topk_scores_BLK, topk_ids_BLK, scores_BLE):
        self._ts, self._tid, self._sc = topk_scores_BLK, topk_ids_BLK, scores_BLE

    def __call__(self, x_BLD, expert_bias_E=None):
        return self._ts, self._tid, self._sc


def run_moe_counts(moe, *, topk_ids_BLK, num_experts, B, L, D, ncalls=1):
    """Run scope MoE.forward `ncalls` times; return (last captured counts, tokens_per_expert)."""
    M = make_moe(moe, num_experts=num_experts)
    K = topk_ids_BLK.shape[-1]
    ts = torch.ones(B, L, K)
    sc = torch.zeros(B, L, num_experts)
    M.router = _FixedRouter(ts, topk_ids_BLK, sc)
    M.experts = _FakeExperts(B, L, D)
    x = torch.randn(B, L, D)
    for _ in range(ncalls):
        M.forward(x)
    return M.experts.captured, M.tokens_per_expert_E.clone()


def make_gptoss_experts(go, *, num_experts, dim, hidden_dim, mlp1_w, mlp1_b, mlp2_w, mlp2_b,
                        swiglu_limit=7.0):
    E = go.GptOssGroupedExperts.__new__(go.GptOssGroupedExperts)
    E.num_experts = num_experts
    E.swiglu_limit = swiglu_limit
    E.mlp1_weight_EGD = mlp1_w
    E.mlp1_bias_EG = mlp1_b
    E.mlp2_weight_EDF = mlp2_w
    E.mlp2_bias_ED = mlp2_b
    return E


# --------------------------------------------------------------------------------------
# Hidden cases.
# --------------------------------------------------------------------------------------
def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _weights(seed, E, D, F, scale=0.4):
    g = _gen(seed)
    mlp1_w = torch.randn(E, 2 * F, D, generator=g) * scale
    mlp1_b = torch.randn(E, 2 * F, generator=g) * 0.5
    mlp2_w = torch.randn(E, D, F, generator=g) * scale
    mlp2_b = torch.randn(E, D, generator=g) * 0.5
    return mlp1_w, mlp1_b, mlp2_w, mlp2_b


def _routed(seed, R, D, scale=6.0):
    # large magnitude so h_RG spans beyond +/-limit (exercises the clamp point).
    return torch.randn(R, D, generator=_gen(seed)) * scale


def allclose(a, b, atol=3e-2, rtol=3e-2):
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))


def build_cases(moe, go):
    C = []

    def add(cid, point, w, thunk):
        C.append((cid, point, w, thunk))

    # ---------- PRODUCER: MoE.forward routing bookkeeping (moe.py) ----------
    def c_counts_topk1():
        # BASE: top_k=1 -> the "first-only" degrade equals correct; passes in ALL modes.
        tid = torch.tensor([[[0], [3], [3], [1], [2]]])  # (1,5,1)
        counts, _ = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=6, B=1, L=5, D=4)
        return torch.equal(counts, ref_counts(tid, 6))
    add("counts_topk1_base", "base", 2, c_counts_topk1)

    def c_counts_topk2():
        # DISCRIMINATIVE P1: top_k=2 -> must count BOTH experts per token.
        tid = torch.tensor([[[0, 3], [3, 5], [1, 3], [2, 0], [4, 1]]])  # (1,5,2)
        counts, _ = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=6, B=1, L=5, D=4)
        return torch.equal(counts, ref_counts(tid, 6))
    add("counts_topk2_a", "P1", 3, c_counts_topk2)

    def c_counts_topk2_b():
        tid = torch.tensor([[[2, 5], [2, 5], [0, 1], [3, 4], [2, 3], [5, 0]]])  # (1,6,2)
        counts, _ = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=6, B=1, L=6, D=4)
        return torch.equal(counts, ref_counts(tid, 6))
    add("counts_topk2_b", "P1", 2, c_counts_topk2_b)

    def c_tpe_one_call():
        # BASE: after ONE forward, copy_ and add_ from a zeroed buffer agree; passes ALL modes.
        tid = torch.tensor([[[0, 3], [3, 5], [1, 3], [2, 0], [4, 1]]])
        _, tpe = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=6, B=1, L=5, D=4, ncalls=1)
        return torch.equal(tpe, ref_counts(tid, 6))
    add("tpe_one_call_base", "base", 2, c_tpe_one_call)

    def c_tpe_two_calls():
        # DISCRIMINATIVE P2: after TWO forwards the counter must ACCUMULATE (2x).
        tid = torch.tensor([[[0, 3], [3, 5], [1, 3], [2, 0], [4, 1]]])
        _, tpe = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=6, B=1, L=5, D=4, ncalls=2)
        return torch.equal(tpe, 2 * ref_counts(tid, 6))
    add("tpe_two_calls_accumulate", "P2", 3, c_tpe_two_calls)

    # ---------- CONSUMER: swiglu (gpt_oss/moe.py) ----------
    def c_swiglu_split():
        # DISCRIMINATIVE P3: interleaved even/odd split (moderate magnitude, no clamp).
        x = torch.randn(4, 8, generator=_gen(21)) * 2.0
        return allclose(go.swiglu(x, limit=7.0), ref_swiglu(x, limit=7.0))
    add("swiglu_interleaved_split", "P3", 3, c_swiglu_split)

    def c_swiglu_bias1():
        # DISCRIMINATIVE P4: the +1 bias on the linear branch. Use interleaved-symmetric
        # input (even==odd) so the split (P3) is a no-op and only P4 separates.
        half = torch.randn(4, 4, generator=_gen(22)) * 1.5
        x = torch.repeat_interleave(half, 2, dim=-1)  # x[::2]==x[1::2]==half channels
        return allclose(go.swiglu(x, limit=7.0), ref_swiglu(x, limit=7.0))
    add("swiglu_linear_bias_of_one", "P4", 3, c_swiglu_bias1)

    def c_swiglu_clamp():
        # DISCRIMINATIVE P5: gate values below -limit. glu must NOT be min-clamped.
        half = torch.full((3, 4), -12.0)  # < -limit on every gate channel
        x = torch.repeat_interleave(half, 2, dim=-1)
        return allclose(go.swiglu(x, limit=7.0), ref_swiglu(x, limit=7.0))
    add("swiglu_glu_onesided_clamp", "P5", 2, c_swiglu_clamp)

    # ---------- CONSUMER: full grouped expert compute (gpt_oss/moe.py) ----------
    def _experts_case(seed, counts_list, limit=7.0):
        E = len(counts_list); D = 4; F = 3
        counts = torch.tensor(counts_list, dtype=torch.int32)
        R = int(counts.sum().item())
        mlp1_w, mlp1_b, mlp2_w, mlp2_b = _weights(seed, E, D, F)
        x = _routed(seed + 100, R, D)
        obj = make_gptoss_experts(go, num_experts=E, dim=D, hidden_dim=F,
                                  mlp1_w=mlp1_w, mlp1_b=mlp1_b, mlp2_w=mlp2_w,
                                  mlp2_b=mlp2_b, swiglu_limit=limit)
        got = obj._experts_forward(x, counts)
        want = ref_experts_forward(mlp1_w, mlp1_b, mlp2_w, mlp2_b, x, counts, limit)
        return allclose(got, want)

    def c_experts_a():
        # exercises P3-P6 together + the offset/bias plumbing (with correct counts fed in).
        return _experts_case(31, [2, 3, 1, 2])
    add("experts_forward_a", "P3P4P5P6", 3, c_experts_a)

    def c_experts_b():
        return _experts_case(32, [1, 1, 4, 2, 2])
    add("experts_forward_b", "P3P4P5P6", 2, c_experts_b)

    # ---------- END-TO-END: couples moe.py (P1 counts) + gpt_oss compute ----------
    def _e2e_case(seed, tid, num_experts):
        B, L, K = tid.shape
        D = 4; F = 3
        # scope counts from MoE.forward (may be degraded by P1)
        counts_scope, _ = run_moe_counts(moe, topk_ids_BLK=tid, num_experts=num_experts,
                                         B=B, L=L, D=D)
        counts_ref = ref_counts(tid, num_experts).to(torch.int32)
        R = int(counts_ref.sum().item())  # routed tokens = sum of CORRECT counts
        mlp1_w, mlp1_b, mlp2_w, mlp2_b = _weights(seed, num_experts, D, F)
        x = _routed(seed + 200, R, D)
        obj = make_gptoss_experts(go, num_experts=num_experts, dim=D, hidden_dim=F,
                                  mlp1_w=mlp1_w, mlp1_b=mlp1_b, mlp2_w=mlp2_w, mlp2_b=mlp2_b)
        got = obj._experts_forward(x, counts_scope.to(torch.int32))
        want = ref_experts_forward(mlp1_w, mlp1_b, mlp2_w, mlp2_b, x,
                                   counts_ref, 7.0)
        return allclose(got, want)

    def c_e2e_a():
        tid = torch.tensor([[[0, 3], [3, 5], [1, 3], [2, 0], [4, 1]]])
        return _e2e_case(41, tid, 6)
    add("e2e_route_then_experts_a", "coupling", 3, c_e2e_a)

    def c_e2e_b():
        tid = torch.tensor([[[2, 5], [2, 5], [0, 1], [3, 4], [2, 3], [5, 0]]])
        return _e2e_case(42, tid, 6)
    add("e2e_route_then_experts_b", "coupling", 3, c_e2e_b)

    def c_e2e_c():
        tid = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
        return _e2e_case(43, tid, 4)
    add("e2e_route_then_experts_c", "coupling", 2, c_e2e_c)

    return C


def run():
    moe, go = load_modules()
    cases = build_cases(moe, go)
    total = sum(w for _, _, w, _ in cases)
    got = 0
    trace = []
    for cid, point, w, thunk in cases:
        try:
            ok = bool(thunk())
        except Exception as e:
            trace.append({"case": cid, "point": point, "weight": w, "ok": False,
                          "err": repr(e)[:200]})
            continue
        if ok:
            got += w
        trace.append({"case": cid, "point": point, "weight": w, "ok": ok})
    score = round(got / total, 6) if total else 0.0
    moe_file = getattr(moe, "__file__", "")
    go_file = getattr(go, "__file__", "")
    origin_ok = moe_file.startswith(REPO) and go_file.startswith(REPO)
    return {"module": moe_file if origin_ok else "",
            "module_moe": moe_file, "module_gptoss": go_file,
            "origin_ok": origin_ok, "score": score, "got": got,
            "total": total, "cases": trace}


def main():
    try:
        r = run()
    except Exception as e:
        import traceback
        print("WRO_RESULT " + json.dumps({"module": "", "score": 0.0,
              "error": repr(e), "tb": traceback.format_exc()[-1200:]}))
        return
    print("WRO_RESULT " + json.dumps(r))


if __name__ == "__main__":
    main()
