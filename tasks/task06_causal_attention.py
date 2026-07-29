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
        self.D = model_dim
        self.H = num_heads
        self.Dh = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3*model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim, bias=False)

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
        H, Dh = self.H, self.Dh
        qkv: torch.Tensor = self.qkv(x) # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1) # [B, T, D] each
        q = q.reshape(B, T, H, Dh).transpose(-2,-3) # [B, H, T, Dh]
        k = k.reshape(B, T, H, Dh).transpose(-2,-3) # [B, H, T, Dh]
        v = v.reshape(B, T, H, Dh).transpose(-2,-3) # [B, H, T, Dh]
        attn = q @ k.transpose(-1,-2) * (Dh ** -0.5) # [B, H, T, T]
        mask = torch.tril(
            torch.ones(size=(T,T), dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        if padding_mask is not None:
            mask = mask & padding_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
            # 1 1 0
            # 1 1 0
            # 1 1 0
            # row softmaxes still ok after erasing column
        attn = torch.masked_fill(attn, ~mask, float("-inf")) # [B, H, T, T]
        probs = F.softmax(attn, dim=-1) # [B, H, T, T]
        probs = self.dropout(probs)
        attn_out = probs @ v # [B, H, T, Dh]
        attn_out = attn_out.transpose(-2,-3).reshape(B, T, D)
        if padding_mask is not None:
            attn_out = torch.masked_fill(
                attn_out,
                ~padding_mask.unsqueeze(-1), # [B, T, 1]
                0,
            )
        return self.out(attn_out)
