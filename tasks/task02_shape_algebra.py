"""Task 2: broadcasting, batched matrix multiplication, heads, and strides."""

from __future__ import annotations

import torch


def pairwise_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute pairwise dot products, cosine similarity, and squared distance.

    Args:
        x: Tensor with shape ``[B, N, D]``.
        y: Tensor with shape ``[B, M, D]``.

    Returns:
        ``(dot_products, cosine_similarities, squared_distances)``, each with
        shape ``[B, N, M]``.

    Constraints:
        Do not use Python loops, ``torch.cdist``, ``repeat``, or
        ``repeat_interleave``.
    """
    dot_products = x @ y.transpose(-1,-2) # [B, N, M]
    x_sq_norms = x.square().sum(dim=-1, keepdim=True) # [B, N, 1]
    y_sq_norms = y.square().sum(dim=-1, keepdim=True).transpose(-1,-2) # [B, 1, M]
    cosine_similarities = dot_products * torch.rsqrt(x_sq_norms) * torch.rsqrt(y_sq_norms)
    squared_distances = x_sq_norms + y_sq_norms - 2 * dot_products
    return (dot_products, cosine_similarities, squared_distances)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert ``[B, T, H*Dh]`` to ``[B, H, T, Dh]``."""
    B, T, _ = x.shape
    return x.reshape(B, T, num_heads, -1).transpose(-2,-3)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, H, T, Dh]`` to ``[B, T, H*Dh]``."""
    B, _, T, _ = x.shape
    return x.transpose(-2,-3).reshape(B, T, -1)
