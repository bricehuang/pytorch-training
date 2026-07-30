"""Task 7: LayerNorm implemented with a custom autograd Function."""

from __future__ import annotations

import torch
from torch import nn


class ManualLayerNormFunction(torch.autograd.Function):
    """Custom final-dimension LayerNorm with an explicitly derived backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        y = x - x.mean(dim=-1, keepdim=True)
        var = y.square().mean(dim=-1, keepdim=True)
        inv_stdev = (var + eps).rsqrt()
        # save stuff to ctx
        ctx.save_for_backward(y, inv_stdev, weight)
        return y * inv_stdev * weight + bias

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        y, inv_stdev, w = ctx.saved_tensors
        x_normalized = y * inv_stdev
        batch_dims = tuple(range(grad_output.ndim-1))
        grad_y1 = grad_output * inv_stdev * w
        grad_y2 = (grad_output * x_normalized * w).mean(dim=-1, keepdim=True) * inv_stdev.square() * y
        grad_y = grad_y1 - grad_y2
        grad_x = grad_y - grad_y.mean(dim=-1, keepdim=True)
        grad_w = (grad_output * x_normalized).sum(dim=batch_dims)
        grad_b = grad_output.sum(dim=batch_dims)
        return (grad_x, grad_w, grad_b, None)


class ManualLayerNorm(nn.Module):
    """Learnable LayerNorm over the final input dimension."""

    def __init__(self, normalized_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ManualLayerNormFunction.apply(x, self.weight, self.bias, self.eps)
