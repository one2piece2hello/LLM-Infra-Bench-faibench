# Reviewer-only ORACLE (not baked into the image): fused single-pass all-reduce + bias. Used only
# to (a) calibrate the oracle_ms latency constant on the authoring lane and (b) prove the correctness + headroom
# gradient. Grounded in SERVE.PARALLEL.TP: nano-vllm RowParallelLinear.forward (linear.py:131,
# all_reduce(y)), litgpt all_reduce_output (generate/tp.py:78), and TGI TensorParallelRowLinear
# (tensor_parallel.py:202) — the row-parallel outputs are summed across ranks and the bias is
# added once. torch reduces the whole [R, T, D] over the rank axis in ONE fused pass (fp32
# accumulate) and writes the [T, D] result once -> far less HBM traffic and one launch, vs the
# per-rank python accumulate (baseline2). The gap grows with R (the tensor-parallel world size).
import torch


def reduce_partials(partials, bias):
    acc = partials.sum(dim=0, dtype=torch.float32)   # fused reduction over the rank axis (fp32)
    acc = acc + bias.float()                          # bias added once (broadcast over T rows)
    return acc.to(torch.bfloat16)


def custom_kernel(data):
    partials, bias, config = data
    return reduce_partials(partials, bias)
