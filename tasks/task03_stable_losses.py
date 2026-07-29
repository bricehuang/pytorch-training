"""Task 3: numerically stable softmax-family operations."""

from __future__ import annotations

import torch


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log-softmax without PyTorch's softmax/logsumexp helpers."""
    logits = logits - logits.amax(dim=dim, keepdim=True) # [..., D]
    log_normalization = torch.log(
        torch.exp(logits).sum(dim=dim,keepdim=True)
    )
    return logits - log_normalization

def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute softmax over allowed entries.

    Args:
        scores: Input scores.
        mask: Boolean tensor broadcastable to ``scores``. ``True`` means the
            element is allowed. Every reduction row contains at least one
            allowed element.
        dim: Softmax dimension.
    """
    logits = torch.masked_fill(scores, ~mask, float("-inf"))
    return torch.exp(stable_log_softmax(logits, dim))


def cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean cross-entropy for logits ``[B, C]`` and targets ``[B]``.

    Do not use PyTorch's softmax, log-softmax, logsumexp, or cross-entropy
    implementations inside this function.
    """
    log_softmax = stable_log_softmax(logits, dim=-1)
    entropies = torch.gather(
        log_softmax,
        dim=-1,
        index=targets.unsqueeze(-1)
    )
    return -entropies.mean()
