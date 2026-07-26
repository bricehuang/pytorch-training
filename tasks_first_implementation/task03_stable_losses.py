"""Task 3: numerically stable softmax-family operations."""

from __future__ import annotations

import torch


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log-softmax without PyTorch's softmax/logsumexp helpers."""
    shift_logits = logits - logits.amax(dim=dim, keepdim=True)
    numerator = torch.exp(shift_logits)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return shift_logits - torch.log(denominator)


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
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    shift_logits = masked_scores - masked_scores.amax(dim=dim, keepdim=True)
    numerator = torch.exp(shift_logits)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator


def cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean cross-entropy for logits ``[B, C]`` and targets ``[B]``.

    Do not use PyTorch's softmax, log-softmax, logsumexp, or cross-entropy
    implementations inside this function.
    """
    shift_logits = logits - logits.amax(dim=-1, keepdim=True) # [B, C]
    logsumexp = torch.log(torch.exp(shift_logits).sum(dim=-1))
    B = targets.shape[0]
    indexed_logits = shift_logits[torch.arange(B), targets]
    return (logsumexp - indexed_logits).mean()
