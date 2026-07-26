"""Task 8: complete manual backpropagation through a two-layer MLP."""

from __future__ import annotations

import torch


def manual_mlp_loss_and_gradients(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MLP cross-entropy loss and parameter gradients by hand.

    Network:
        ``Z1 = X @ W1 + b1``
        ``H = tanh(Z1)``
        ``Z2 = H @ W2 + b2``
        ``loss = mean_cross_entropy(Z2, y)``

    Shapes:
        - ``X``: ``[B, Din]``
        - ``y``: ``[B]``
        - ``W1``: ``[Din, H]``
        - ``b1``: ``[H]``
        - ``W2``: ``[H, C]``
        - ``b2``: ``[C]``

    Constraints:
        Inputs have ``requires_grad=False``. Do not call ``backward`` or
        ``torch.autograd.grad`` and do not construct differentiable temporary
        tensors. Return gradients under keys ``W1``, ``b1``, ``W2``, ``b2``.
    """
    raise NotImplementedError("Implement manual_mlp_loss_and_gradients")
