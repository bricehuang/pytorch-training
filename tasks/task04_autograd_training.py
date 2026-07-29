"""Task 4: raw tensors, autograd, and a manual optimization loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def train_logistic_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    learning_rate: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Train binary logistic regression using raw parameter tensors.

    Args:
        X: Feature matrix with shape ``[N, D]``.
        y: Binary labels with shape ``[N]``. Labels may be integer or floating
            point but contain only zero and one.
        learning_rate: SGD step size.
        steps: Number of full-batch optimization steps.

    Returns:
        ``(weight, bias, loss_history)`` where ``weight`` has shape ``[D, 1]``,
        ``bias`` has shape ``[1]``, and the history has length ``steps``.

    Constraints:
        Use raw tensors with ``requires_grad=True``, ``loss.backward()``,
        updates under ``torch.no_grad()``, and explicit gradient clearing.
        Do not use ``nn.Module`` or ``torch.optim``.
    """
    _, D = X.shape
    w = torch.zeros(size=(D,1), requires_grad=True, dtype=X.dtype, device=X.device)
    b = torch.zeros(size=(1,), requires_grad=True, dtype=X.dtype, device=X.device)
    y = y.long()
    loss_history = []
    for _ in range(steps):
        w.grad = None
        b.grad = None
        scores = X @ w + b # [N,1]
        logits = torch.cat(
            (torch.zeros_like(scores), scores),
            dim=-1,
        )
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
        loss_history.append(loss.item())
    return w, b, loss_history
