"""REVIEWER-ONLY negative: a degenerate constant-logit stub.

Proves the anti-spoof gate fires (the design principle "known-bad = 0 for its NAMED
reason"): it returns a module whose logits do not depend on the input at all, so the
position-variance probe and the real-vs-arbitrary OOD probe must reject it and the reward
must be exactly 0 even though the artifact is tiny (well inside the byte budget).

NEVER model-visible: baked root-0700 at /opt/negative/.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConstantLM(nn.Module):
    def __init__(self, vocab_size: int, n_pad_params: int = 320_000_000):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        # pad the parameter count so it clears the min_param_count floor and the ONLY
        # failing gates are the anti-spoof probes (a sharper negative).
        self.pad = nn.Parameter(torch.zeros(max(n_pad_params // 1024, 1), 1024), requires_grad=False)

    def forward(self, input_ids, labels=None, **kw):
        b, t = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(b, t, self.vocab_size).contiguous()
        return type("O", (), {"logits": logits, "loss": None})()


def build_adapted_model(base_model, artifact_path, device):
    vocab = int(getattr(base_model.config, "vocab_size", 151936))
    del base_model
    torch.cuda.empty_cache()
    return ConstantLM(vocab).to(device).eval()
