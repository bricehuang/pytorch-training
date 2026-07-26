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
        raise NotImplementedError("Implement ManualLayerNormFunction.forward")

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        raise NotImplementedError("Implement ManualLayerNormFunction.backward")


class ManualLayerNorm(nn.Module):
    """Learnable LayerNorm over the final input dimension."""

    def __init__(self, normalized_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        raise NotImplementedError("Implement ManualLayerNorm.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement ManualLayerNorm.forward")
