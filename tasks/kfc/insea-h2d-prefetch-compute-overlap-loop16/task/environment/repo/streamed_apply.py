"""Stream a list of host-resident chunks to the GPU and apply a per-chunk op.

Public entry point:
    ``streamed_chunk_apply(chunks, compute) -> torch.Tensor``

This is the data-movement step in front of a chunked GPU workload: a large tensor
lives in host (CPU) memory, already split into ``N`` chunks (row-blocks). Each chunk
must be moved host-to-device and then transformed by a per-chunk ``compute`` op that
runs on the GPU. The processed chunks are concatenated (in input order) into a single
device tensor.

Contract
--------
- ``chunks``: a ``list``/``tuple`` of CPU ``torch.Tensor`` (host memory). Each chunk is
  at least 1-D; the leading dimension is the row/block axis and may differ across
  chunks, while every chunk shares the same trailing shape ``chunk.shape[1:]`` and the
  same floating dtype.
- ``compute``: a callable mapping a CUDA tensor to a CUDA tensor. It is applied once to
  the on-device copy of each chunk and preserves the leading (row) dimension. Treat it
  as an opaque GPU operation — invoke it once per chunk; do not inspect or rewrite it.

Functionality::

    for each chunk c (in order):
        d       = c copied to the GPU            # host -> device transfer
        out_c   = compute(d)                     # per-chunk GPU work
    result = concatenate(out_c for all chunks, along dim 0)

The result is a single CUDA tensor equal to running the transfer-then-compute for
every chunk sequentially and concatenating the outputs in the original chunk order.
An empty ``chunks`` list yields an empty CUDA tensor.

Returns
-------
A CUDA ``torch.Tensor``: the per-chunk outputs concatenated along dim 0, in chunk order.

Error contract
--------------
- ``TypeError`` if ``compute`` is not callable, ``chunks`` is not a list/tuple, a chunk
  is not a ``torch.Tensor``, or the chunks do not all share one dtype.
- ``ValueError`` if a chunk is not in host (CPU) memory, a chunk is 0-D, or the chunks
  do not all share the same trailing shape ``chunk.shape[1:]``.

Note on allowed operations
--------------------------
Build the transfer/compute schedule yourself with explicit CUDA stream and event
management (``torch.cuda.Stream`` / ``torch.cuda.Event`` / pinned host memory /
non-blocking copies are the intended tools). Do NOT delegate the pipelining to a
framework auto-overlap / graph-capture convenience (CUDA-graph capture-and-replay of
the loop, dataloader-style background prefetchers, or any helper that auto-pipelines
host-to-device copies with compute); those are out of scope and are blocked at scoring.

The current implementation moves one chunk to the GPU and only then launches its
compute, one chunk fully after another on a single stream. While a chunk's bytes are
streaming in, the GPU compute units sit idle; while a chunk is being computed, the
transfer engine sits idle. The two phases never run at the same time, so the total
latency is the *sum* of all transfers plus all compute — the transfer latency is fully
exposed instead of being hidden behind compute.
"""

import torch


def _validate(chunks, compute):
    if not callable(compute):
        raise TypeError("compute must be callable")
    if not isinstance(chunks, (list, tuple)):
        raise TypeError(f"chunks must be a list or tuple, got {type(chunks)}")
    ref_trailing = None
    ref_dtype = None
    for i, c in enumerate(chunks):
        if not isinstance(c, torch.Tensor):
            raise TypeError(f"chunk {i} must be a torch.Tensor, got {type(c)}")
        if c.is_cuda:
            raise ValueError(f"chunk {i} must live in host (CPU) memory, got a CUDA tensor")
        if c.dim() < 1:
            raise ValueError(f"chunk {i} must be at least 1-D, got {c.dim()}-D")
        trailing = tuple(c.shape[1:])
        if ref_trailing is None:
            ref_trailing = trailing
            ref_dtype = c.dtype
        else:
            if trailing != ref_trailing:
                raise ValueError(
                    f"chunk {i} trailing shape {trailing} must equal {ref_trailing}")
            if c.dtype != ref_dtype:
                raise TypeError(f"chunk {i} dtype {c.dtype} must equal {ref_dtype}")


def streamed_chunk_apply(chunks, compute):
    """See module docstring for the full contract.

    Naive serial schedule: for each chunk, copy it to the GPU and then compute it,
    one chunk fully after another on the default stream. Correct, but the host-to-device
    transfer latency is never hidden behind compute (transfers and compute never overlap),
    so total latency is the sum of every transfer plus every compute.
    """
    _validate(chunks, compute)
    if len(chunks) == 0:
        return torch.empty(0, device="cuda")
    outs = []
    for c in chunks:
        d = c.to("cuda", non_blocking=False)   # blocking host -> device on the default stream
        outs.append(compute(d))                # compute only starts after the copy finishes
    return torch.cat(outs, dim=0)
