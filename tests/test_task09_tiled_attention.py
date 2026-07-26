import math

import torch

from tasks.task09_tiled_attention import tiled_attention


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        tq, tk = scores.shape[-2:]
        mask = torch.arange(tk, device=q.device).view(1, tk) <= torch.arange(
            tq, device=q.device
        ).view(tq, 1)
        scores = scores.masked_fill(~mask, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v


def test_noncausal_tiled_attention_matches_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 9, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 9, 4, dtype=torch.float64)

    actual = tiled_attention(
        q,
        k,
        v,
        query_block_size=3,
        key_block_size=4,
        causal=False,
    )
    expected = reference_attention(q, k, v, causal=False)
    assert actual.shape == (2, 3, 7, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_causal_tiled_attention_matches_reference() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 7, 4, dtype=torch.float64) * 3
    k = torch.randn(1, 2, 7, 4, dtype=torch.float64) * 3
    v = torch.randn(1, 2, 7, 6, dtype=torch.float64)

    actual = tiled_attention(
        q,
        k,
        v,
        query_block_size=4,
        key_block_size=3,
        causal=True,
    )
    expected = reference_attention(q, k, v, causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_different_block_sizes_produce_same_result() -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 1, 6, 3)
    k = torch.randn(1, 1, 8, 3)
    v = torch.randn(1, 1, 8, 2)
    first = tiled_attention(q, k, v, query_block_size=2, key_block_size=3)
    second = tiled_attention(q, k, v, query_block_size=5, key_block_size=7)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
