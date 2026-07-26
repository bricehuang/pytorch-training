"""Task 6: multi-head causal self-attention using ordinary tensor operations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        if model_dim % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide model_dim={model_dim}")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(in_features=model_dim, out_features=3*model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)

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
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        qkv = self.qkv(x) # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1) # [B, T, D] each
        q = q.reshape(B, T, H, Dh).transpose(1,2) # [B, H, T, Dh]
        k = k.reshape(B, T, H, Dh).transpose(1,2) # [B, H, T, Dh]
        v = v.reshape(B, T, H, Dh).transpose(1,2) # [B, H, T, Dh]
        mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0) # [1, 1, T, T] bool
        if padding_mask is not None:
            padding = padding_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T] bool
            mask = mask & padding
        a = q @ k.transpose(2,3) * Dh ** -0.5 # [B, H, T, T]
        a = a.masked_fill(~mask, float("-inf"))
        s = F.softmax(a, dim=-1) # [B, H, T, T]
        s = self.dropout(s)
        o = s @ v # [B, H, T, Dh]
        attn_out = o.transpose(1,2).reshape(B, T, D)
        proj_out = self.out(attn_out) # [B, T, D]
        if padding_mask is not None:
            out_mask = padding_mask.unsqueeze(2) # [B, T, 1]
            proj_out = proj_out.masked_fill(~out_mask, 0.0)
        return proj_out
