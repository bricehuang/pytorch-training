"""Task 10: FlashAttention-style tiled forward and manually derived backward."""

from __future__ import annotations

import torch


class FlashAttentionFunction(torch.autograd.Function):
    """Custom exact attention operation that avoids saving an O(T^2) matrix."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_block_size: int,
        key_block_size: int,
        causal: bool,
    ) -> torch.Tensor:
        """Run tiled online-softmax attention and save compact backward state."""
        raise NotImplementedError("Implement FlashAttentionFunction.forward")

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        """Recompute probability tiles and return ``dq``, ``dk``, and ``dv``."""
        raise NotImplementedError("Implement FlashAttentionFunction.backward")


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_block_size: int = 64,
    key_block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """User-facing wrapper for :class:`FlashAttentionFunction`.

    Inputs have shapes ``q=[B,H,Tq,D]``, ``k=[B,H,Tk,D]``, and
    ``v=[B,H,Tk,Dv]``. Public causal tests use ``Tq == Tk``.
    """
    raise NotImplementedError("Implement flash_attention")
