import math

import torch

from tasks.task10_flash_attention import flash_attention


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        length = scores.shape[-1]
        mask = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v


def compare_output_and_gradients(*, causal: bool) -> None:
    torch.manual_seed(4 if causal else 3)
    length_q = 5
    length_k = 5 if causal else 7

    q_actual = torch.randn(1, 2, length_q, 4, dtype=torch.float64, requires_grad=True)
    k_actual = torch.randn(1, 2, length_k, 4, dtype=torch.float64, requires_grad=True)
    v_actual = torch.randn(1, 2, length_k, 3, dtype=torch.float64, requires_grad=True)
    q_expected = q_actual.detach().clone().requires_grad_(True)
    k_expected = k_actual.detach().clone().requires_grad_(True)
    v_expected = v_actual.detach().clone().requires_grad_(True)

    actual = flash_attention(
        q_actual,
        k_actual,
        v_actual,
        query_block_size=3,
        key_block_size=2,
        causal=causal,
    )
    expected = reference_attention(q_expected, k_expected, v_expected, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(q_actual.grad, q_expected.grad, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(k_actual.grad, k_expected.grad, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(v_actual.grad, v_expected.grad, rtol=1e-8, atol=1e-9)


def test_noncausal_output_and_gradients() -> None:
    compare_output_and_gradients(causal=False)


def test_causal_output_and_gradients() -> None:
    compare_output_and_gradients(causal=True)


def test_gradcheck_tiny_case() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda q_, k_, v_: flash_attention(
            q_,
            k_,
            v_,
            query_block_size=2,
            key_block_size=2,
            causal=True,
        ),
        (q, k, v),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
