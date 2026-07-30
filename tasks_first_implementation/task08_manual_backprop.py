"""Task 8: complete manual backpropagation through a two-layer MLP."""

from __future__ import annotations

import torch

import torch.nn.functional as F

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
    # forward pass
    B, _ = X.shape
    _, C = W2.shape
    Z1 = X @ W1 + b1 # [B, H]
    H = torch.tanh(Z1) # [B, H]
    Z2 = H @ W2 + b2 # [B, C]
    loss = F.cross_entropy(Z2, y)
    # backward pass
    grad_Z2_part1 = F.softmax(Z2, dim=-1) 
    # y_unsqueezed = y.unsqueeze(-1)
    # grad_Z2_part2 = torch.zeros_like(Z2)
    # grad_Z2_part2.scatter_(
    #     dim=-1,
    #     index=y_unsqueezed,
    #     src=torch.ones_like(y_unsqueezed, dtype=Z2.dtype)
    # )
    grad_Z2_part2 = F.one_hot(y, num_classes=C)
    grad_Z2 = (grad_Z2_part1 - grad_Z2_part2) / B # [B, C]
    grad_H  = grad_Z2 @ W2.transpose(-1,-2) # [B, H]
    grad_W2 = H.transpose(-1,-2) @ grad_Z2 # [H, C]
    grad_b2 = grad_Z2.sum(0) # [C]
    grad_Z1 = (1 - H.square()) * grad_H # [B, H]
    grad_W1 = X.transpose(-1,-2) @ grad_Z1 # [Din, H]
    grad_b1 = grad_Z1.sum(0) # [H]

    grads = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }
    return (loss, grads)
