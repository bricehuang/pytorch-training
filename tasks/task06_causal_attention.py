"""Task 6: multi-head causal self-attention using ordinary tensor operations."""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    ``padding_mask`` uses ``True`` for valid tokens and ``False`` for padding.
    Do not use ``nn.MultiheadAttention`` or
    ``torch.nn.functional.scaled_dot_product_attention``.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Implement CausalSelfAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Tensor with shape ``[B, T, D]``.
            padding_mask: Optional Boolean tensor with shape ``[B, T]``;
                ``True`` denotes a valid token.

        Returns:
            Tensor with shape ``[B, T, D]``.
        """
        raise NotImplementedError("Implement CausalSelfAttention.forward")
