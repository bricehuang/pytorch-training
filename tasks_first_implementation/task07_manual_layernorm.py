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
        var, mean = torch.var_mean(x, dim=-1, correction=0, keepdim=True)
        x_centered = x - mean
        var_rescale = torch.rsqrt(var + eps)
        x_normalized = x_centered * var_rescale
        out = x_normalized * weight + bias
        # save stuff to ctx
        ctx.save_for_backward(x_centered, var_rescale, weight)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        batch_dims = tuple(range(grad_output.ndim - 1))
        x_centered, var_rescale, weight = ctx.saved_tensors
        x_normalized = x_centered * var_rescale
        grad_x_centered_diagonal = grad_output * weight * var_rescale
        grad_x_centered_cross1 = x_centered * var_rescale.pow(3)
        grad_x_centered_cross2 = (grad_output * x_centered * weight).mean(dim=-1, keepdim=True)
        grad_x_centered = grad_x_centered_diagonal - grad_x_centered_cross1 * grad_x_centered_cross2
        grad_x = grad_x_centered - grad_x_centered.mean(dim=-1, keepdim=True)
        grad_bias = grad_output.sum(dim=batch_dims)
        grad_weight = (grad_output * x_normalized).sum(dim=batch_dims)
        return (grad_x, grad_weight, grad_bias, None)


class ManualLayerNorm(nn.Module):
    """Learnable LayerNorm over the final input dimension."""

    def __init__(self, normalized_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((normalized_dim,)))
        self.bias = nn.Parameter(torch.zeros((normalized_dim,)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ManualLayerNormFunction.apply(x, self.weight, self.bias, self.eps)
