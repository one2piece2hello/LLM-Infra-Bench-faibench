"""Cross-entropy loss and input-gradient over a batch of logit rows.

Public entry point:
    ``cross_entropy_loss_grad(logits, labels, ignore_index=-100,
                              label_smoothing=0.0) -> (loss, grad)``

Given a batch of ``N`` rows of class scores (``logits``) and one integer target
class per row (``labels``), compute the mean cross-entropy loss over the
non-ignored rows and the gradient of that loss with respect to ``logits``.

Contract
--------
- ``logits``: shape ``(N, V)``, a floating-point CUDA tensor (``float32`` or
  ``bfloat16``). ``N`` is the number of rows (batch * sequence), ``V`` the
  number of classes (vocabulary size).
- ``labels``: shape ``(N,)``, an integer CUDA tensor. Each entry is a class
  index in ``[0, V)``, or equals ``ignore_index`` to mark the row as ignored.
- ``ignore_index``: int. Rows whose label equals this value contribute zero to
  the loss and zero to the gradient.
- ``label_smoothing``: float ``>= 0``. ``0.0`` is ordinary cross entropy.

Returns
-------
- ``loss``: a scalar tensor — the mean cross-entropy (with optional label
  smoothing) over the non-ignored rows. If every row is ignored, the loss is
  ``0``.
- ``grad``: shape ``(N, V)``, same dtype as ``logits`` — the gradient of
  ``loss`` w.r.t. ``logits``. Ignored rows are exactly zero.

Definition (per non-ignored row with target ``y``, over ``n`` non-ignored rows):
    ``p = softmax(logits)``
    ``loss_row = -(1 - s) * log p[y] - (s / V) * sum_k log p[k]``
    ``grad_row[j] = ( p[j] - s / V - (1 - s) * [j == y] ) / n``
where ``s = label_smoothing``. All reductions are computed in float32; the
gradient is returned in the input dtype.

Error contract
--------------
- ``TypeError`` if ``logits`` is not a floating-point tensor, or ``labels`` is
  not an integer tensor.
- ``ValueError`` if ``logits`` is not 2-D; if ``labels`` is not 1-D of length
  ``N``; if ``label_smoothing`` is negative; or if any non-ignored label lies
  outside ``[0, V)``.

Note on the current implementation
----------------------------------
This straightforward implementation forms the full log-probability tensor, the
full probability tensor, and a full gradient tensor -- three ``(N, V)`` buffers
in addition to the input. It is correct but holds several full-size copies of
the logits at once, which is expensive at large ``V``.
"""

import torch


def _validate(logits, labels, label_smoothing):
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensor")
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be a floating-point tensor, got {logits.dtype}")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be an integer tensor, got {labels.dtype}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (N, V), got {logits.dim()}-D")
    N, V = logits.shape
    if labels.dim() != 1 or labels.shape[0] != N:
        raise ValueError(f"labels must be 1-D of length N={N}, got shape {tuple(labels.shape)}")
    if float(label_smoothing) < 0.0:
        raise ValueError(f"label_smoothing must be >= 0, got {label_smoothing}")
    return N, V


def cross_entropy_loss_grad(logits, labels, ignore_index=-100, label_smoothing=0.0):
    """See module docstring for the full contract."""
    N, V = _validate(logits, labels, label_smoothing)
    s = float(label_smoothing)

    valid = labels != ignore_index
    # non-ignored labels must be inside [0, V)
    if valid.any():
        vlab = labels[valid]
        if int(vlab.max()) >= V or int(vlab.min()) < 0:
            raise ValueError(f"a non-ignored label is out of range [0, {V})")
    n_valid = int(valid.sum().item())
    denom = max(n_valid, 1)

    safe = labels.clamp(min=0).long()                       # ignore rows point at col 0 (masked out later)
    idx = torch.arange(N, device=logits.device)

    lf = logits.to(torch.float32)
    logp = torch.log_softmax(lf, dim=-1)                    # (N, V) full buffer
    logp_y = logp[idx, safe]                                # (N,)
    nll = -logp_y
    smooth = -logp.mean(dim=1)                              # -mean_k log p[k]
    row_loss = (1.0 - s) * nll + s * smooth
    row_loss = torch.where(valid, row_loss, torch.zeros_like(row_loss))
    loss = row_loss.sum() / denom

    prob = logp.exp()                                       # (N, V) softmax
    grad = prob.clone()                                     # (N, V) gradient buffer
    if s != 0.0:
        grad = grad - (s / V)
    grad[idx, safe] = grad[idx, safe] - (1.0 - s)
    grad = grad / denom
    grad[~valid] = 0.0                                      # ignored rows -> exactly zero
    return loss, grad.to(logits.dtype)
