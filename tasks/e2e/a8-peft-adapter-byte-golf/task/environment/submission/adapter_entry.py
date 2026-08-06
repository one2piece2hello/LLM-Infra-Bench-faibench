"""Reference adapter loader — the frozen ENTRY CONTRACT of this task.

The grader calls exactly one function in this file:

    build_adapted_model(base_model, artifact_path, device) -> nn.Module

`base_model` is a freshly loaded, UNMODIFIED copy of the frozen base model (the grader
loads it itself from its own read-only copy). `artifact_path` points at your
`adapter.bin`. Return the adapted model the grader should evaluate.

At grading time ONLY this file and `adapter.bin` are carried over — nothing else in the
container is readable while `build_adapted_model` runs, and the two files together must
fit the byte budget stated in `instruction.md`. Both files count, so keep the code small
and put your capacity in the artifact.

You may rewrite this file completely, invent your own artifact format, and build the
adapted model however you like — as long as the returned module is a real language model
whose `forward(input_ids)` yields position-dependent logits `(batch, seq, vocab)`.

The format below ("lora_pack_v1") is only the reference implementation: one flat int8 or
bf16 blob plus an index, unpacked into low-rank side-branches on the targeted
`nn.Linear` layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_DT = {"int8": torch.int8, "bf16": torch.bfloat16, "fp16": torch.float16}


class LoRALinear(nn.Module):
    """base(x) + scale * ((x @ A^T) @ B^T);  A: (r, in), B: (out, r)."""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.register_buffer("lora_A", A, persistent=False)
        self.register_buffer("lora_B", B, persistent=False)
        self.scale = float(scale)

    def forward(self, x):  # noqa: D401
        out = self.base(x)
        h = torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A)
        return out + self.scale * torch.nn.functional.linear(h, self.lora_B).to(out.dtype)


def _unpack(blob: torch.Tensor, entry: dict, dtype: torch.dtype) -> torch.Tensor:
    off, numel = int(entry["off"]), int(entry["numel"])
    raw = blob[off:off + numel]
    if raw.dtype == torch.int8:
        t = raw.to(torch.float32) * float(entry.get("scale", 1.0))
    else:
        t = raw.to(torch.float32)
    return t.reshape(tuple(entry["shape"])).to(dtype)


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, p)
    setattr(parent, parts[-1], new)


def build_adapted_model(base_model, artifact_path, device):
    state = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    fmt = state.get("format", "lora_pack_v1")
    if fmt != "lora_pack_v1":
        raise ValueError(f"unsupported adapter format {fmt!r}")
    blob = state["blob"]
    index = state["index"]
    alpha = float(state.get("alpha", 1.0))
    work_dtype = _DT.get(state.get("compute_dtype", "bf16"), torch.bfloat16)

    base_model = base_model.to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)

    for mod_name, pair in index.items():
        try:
            target = base_model.get_submodule(mod_name)
        except AttributeError:
            continue
        if not isinstance(target, nn.Linear):
            continue
        A = _unpack(blob, pair["A"], work_dtype).to(device)
        B = _unpack(blob, pair["B"], work_dtype).to(device)
        r = A.shape[0]
        _set_submodule(base_model, mod_name, LoRALinear(target, A, B, alpha / max(r, 1)))
    return base_model.eval()
