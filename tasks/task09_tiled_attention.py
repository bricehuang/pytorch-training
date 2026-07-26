"""Task 9: exact tiled attention forward pass with online softmax."""

from __future__ import annotations

import torch


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """Compute exact attention without a full ``[Tq, Tk]`` score matrix.

    Args:
        q: Queries with shape ``[B, H, Tq, D]``.
        k: Keys with shape ``[B, H, Tk, D]``.
        v: Values with shape ``[B, H, Tk, Dv]``.
        query_block_size: Number of query rows in a tile.
        key_block_size: Number of key rows in a tile.
        causal: Apply the usual lower-triangular mask. Public causal tests use
            ``Tq == Tk``.

    Returns:
        Output with shape ``[B, H, Tq, Dv]``.

    Constraints:
        Python loops over tiles are allowed. Do not materialize a tensor whose
        query and key dimensions are simultaneously the full ``Tq`` and ``Tk``.
        Maintain a running maximum, denominator, and weighted-value accumulator
        for each query row.
    """
    raise NotImplementedError("Implement tiled_attention")
