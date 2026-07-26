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
    raise NotImplementedError("Implement creation_workout")


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
    raise NotImplementedError("Implement tensor_workout")
