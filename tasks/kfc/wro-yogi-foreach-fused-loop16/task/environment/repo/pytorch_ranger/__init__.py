"""Minimal local shim for the optional `pytorch_ranger` dependency so that
`import torch_optimizer` succeeds in an environment where the upstream
`pytorch_ranger` wheel is not installed. These names are re-exported by the
package __init__ but are NOT part of any editable scope and are never exercised
by the benchmark workloads."""
from torch.optim.optimizer import Optimizer

__all__ = ("Ranger", "RangerQH", "RangerVA")


class Ranger(Optimizer):
    def __init__(self, params, lr=1e-3, *args, **kwargs):
        super().__init__(params, dict(lr=lr))

    def step(self, closure=None):
        raise NotImplementedError(
            "pytorch_ranger is not installed in this environment"
        )


class RangerQH(Ranger):
    pass


class RangerVA(Ranger):
    pass
