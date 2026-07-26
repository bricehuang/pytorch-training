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
    raise NotImplementedError("Implement pairwise_metrics")


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert ``[B, T, H*Dh]`` to ``[B, H, T, Dh]``."""
    raise NotImplementedError("Implement split_heads")


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, H, T, Dh]`` to ``[B, T, H*Dh]``."""
    raise NotImplementedError("Implement merge_heads")
