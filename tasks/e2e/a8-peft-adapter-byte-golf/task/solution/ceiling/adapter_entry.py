"""REVIEWER-ONLY ceiling probe loader: nibble-packed int4 adapters.

Same LoRA side-branch structure as the reference loader, but the packed blob stores TWO
4-bit values per byte, so the SAME byte budget buys 2x the adapter parameters. This is the
documented sub-8-bit-adapter direction (QLoRA/LoftQ/DuQTTA family) and exists only to
MEASURE that the open-ended reward really goes above 1.0 (SKILL the exit criteria).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base = base
        self.register_buffer("lora_A", A, persistent=False)
        self.register_buffer("lora_B", B, persistent=False)
        self.scale = float(scale)

    def forward(self, x):
        out = self.base(x)
        h = F.linear(x.to(self.lora_A.dtype), self.lora_A)
        return out + self.scale * F.linear(h, self.lora_B).to(out.dtype)


def _unpack_int4(blob, entry, dtype):
    nbytes, numel = int(entry["nbytes"]), int(entry["numel"])
    raw = blob[int(entry["off"]): int(entry["off"]) + nbytes].to(torch.int16)
    lo = (raw & 0x0F).to(torch.int16)
    hi = ((raw >> 4) & 0x0F).to(torch.int16)
    vals = torch.stack([lo, hi], dim=1).reshape(-1)[:numel]
    vals = torch.where(vals > 7, vals - 16, vals)
    return (vals.to(torch.float32) * float(entry["scale"])).reshape(tuple(entry["shape"])).to(dtype)


def _set_submodule(root, name, new):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, p)
    setattr(parent, parts[-1], new)


def build_adapted_model(base_model, artifact_path, device):
    state = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    assert state.get("format") == "lora_pack_int4_v1", state.get("format")
    blob, index, alpha = state["blob"], state["index"], float(state.get("alpha", 1.0))
    base_model = base_model.to(device)
    for p in base_model.parameters():
        p.requires_grad_(False)
    for name, pair in index.items():
        try:
            target = base_model.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(target, nn.Linear):
            continue
        A = _unpack_int4(blob, pair["A"], torch.bfloat16).to(device)
        B = _unpack_int4(blob, pair["B"], torch.bfloat16).to(device)
        _set_submodule(base_model, name, LoRALinear(target, A, B, alpha / max(A.shape[0], 1)))
    return base_model.eval()
