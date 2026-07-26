import torch

from tasks.task06_causal_attention import CausalSelfAttention


def test_attention_shape_and_gradients() -> None:
    torch.manual_seed(0)
    module = CausalSelfAttention(model_dim=12, num_heads=3, dropout=0.0)
    x = torch.randn(2, 5, 12, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_future_tokens_do_not_change_earlier_outputs() -> None:
    torch.manual_seed(1)
    module = CausalSelfAttention(model_dim=8, num_heads=2, dropout=0.0).eval()
    original = torch.randn(1, 6, 8)
    changed = original.clone()
    changed[:, 4:] += 100.0

    output_original = module(original)
    output_changed = module(changed)
    torch.testing.assert_close(output_original[:, :4], output_changed[:, :4])


def test_masked_key_does_not_affect_later_valid_tokens() -> None:
    torch.manual_seed(2)
    module = CausalSelfAttention(model_dim=8, num_heads=2, dropout=0.0).eval()
    mask = torch.tensor([[True, False, True, True, True]])
    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()
    x2[:, 1] += 50.0

    output1 = module(x1, padding_mask=mask)
    output2 = module(x2, padding_mask=mask)
    torch.testing.assert_close(output1[:, 2:], output2[:, 2:])
