"""Task 1: basic tensor creation, indexing, reductions, and reshaping."""

from __future__ import annotations

import torch


def creation_workout() -> dict[str, torch.Tensor]:
    """Create representative tensors with common PyTorch constructors.

    Returns:
        A dictionary with the keys ``tensor``, ``zeros``, ``ones``, ``empty``,
        ``arange``, ``linspace``, ``randn``, ``full``, and ``eye``.

    Notes:
        Choose small, sensible shapes. The public tests check the keys and a
        few broad properties, not one exact set of values.
    """
    return {
        "tensor": torch.tensor([1,2]),
        "zeros": torch.zeros(size=(2,2)),
        "ones": torch.ones(size=(6,7)),
        "empty": torch.empty(size=(4,2)),
        "arange": torch.arange(0,10,1),
        "linspace": torch.linspace(0,6,7),
        "randn": torch.randn(size=(4,2)),
        "full": torch.full(size=(3,3,2), fill_value=1),
        "eye": torch.eye(5),
    }


def tensor_workout(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply basic tensor operations to ``x``.

    Args:
        x: Tensor with shape ``[batch, sequence, features]``.

    Returns:
        A dictionary containing:

        - ``last_token``: ``[B, D]``
        - ``even_features``: ``[B, T, ceil(D / 2)]``
        - ``first_two_batches``: ``[min(B, 2), T, D]``
        - ``positive_mask``: Boolean tensor with shape ``[B, T, D]``
        - ``positive_values``: One-dimensional tensor
        - ``sequence_mean``: ``[B, 1, D]``
        - ``feature_max``: ``[B, T]``
        - ``normalized``: ``[B, T, D]``
        - ``doubled_sequence``: ``[B, 2T, D]``
        - ``flattened_tokens``: ``[B*T, D]``

    Constraints:
        Do not use Python loops and do not hard-code input dimensions.
    """
    sequence_mean = x.mean(dim=1,keepdim=True)
    x_centered = x - sequence_mean
    x_variance = (x_centered * x_centered).mean(dim=1, keepdim=True)
    x_stdev_inv = torch.rsqrt(x_variance)
    normalized = x_centered * x_stdev_inv
    return {
        "last_token": x[:,-1,:],
        "even_features": x[:,:,::2],
        "first_two_batches": x[:2,:,:],
        "positive_mask": x > 0,
        "positive_values": x[x>0],
        "sequence_mean": sequence_mean,
        "feature_max": x.amax(dim=2),
        "normalized": normalized,
        "doubled_sequence": torch.cat((x,x), dim=1),
        "flattened_tokens": x.reshape(-1, x.shape[2]),
    }
