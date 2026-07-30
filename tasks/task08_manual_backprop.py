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
    # X : [B, Din]
    # W1 : [Din, H]
    # b1 : [H]
    B = X.shape[0]
    C = b2.shape[0]
    Z1 = X @ W1 + b1 # [B, H]
    V = torch.tanh(Z1) # [B, H]
    # W2: [H, C]
    # b2: [C]
    Z2 = V @ W2 + b2 # [B, C]
    # y: [B] entries in [C]
    loss = F.cross_entropy(Z2, y)
    grad_Z2 = (F.softmax(Z2, dim=-1) - F.one_hot(y, num_classes=C)) / B # [B, C]
    grad_V  = grad_Z2 @ W2.transpose(-1,-2) # [B, H]
    grad_W2 = V.transpose(-1,-2) @ grad_Z2 # [H, C]
    grad_b2 = grad_Z2.sum(dim=0) # [C]
    grad_Z1 = (1 - V.square()) * grad_V # [B, H]
    grad_W1 = X.transpose(-1,-2) @ grad_Z1 # [Din, H]
    grad_b1 = grad_Z1.sum(dim=0) # [H]
    grad = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }
    return (loss, grad)
