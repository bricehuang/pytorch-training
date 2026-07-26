"""Task 4: raw tensors, autograd, and a manual optimization loop."""

from __future__ import annotations

import torch


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
    raise NotImplementedError("Implement train_logistic_regression")
