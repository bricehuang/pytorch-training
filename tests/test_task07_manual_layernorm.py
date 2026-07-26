import torch
import torch.nn.functional as F

from tasks.task07_manual_layernorm import ManualLayerNorm, ManualLayerNormFunction


def test_forward_and_backward_match_pytorch() -> None:
    torch.manual_seed(0)
    x_actual = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)
    weight_actual = torch.randn(5, dtype=torch.float64, requires_grad=True)
    weight_expected = weight_actual.detach().clone().requires_grad_(True)
    bias_actual = torch.randn(5, dtype=torch.float64, requires_grad=True)
    bias_expected = bias_actual.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(x_actual)

    actual = ManualLayerNormFunction.apply(x_actual, weight_actual, bias_actual, 1e-5)
    expected = F.layer_norm(x_expected, (5,), weight_expected, bias_expected, 1e-5)
    torch.testing.assert_close(actual, expected)

    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(x_actual.grad, x_expected.grad)
    torch.testing.assert_close(weight_actual.grad, weight_expected.grad)
    torch.testing.assert_close(bias_actual.grad, bias_expected.grad)


def test_gradcheck() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b, c: ManualLayerNormFunction.apply(a, b, c, 1e-5),
        (x, weight, bias),
    )


def test_module_supports_noncontiguous_input() -> None:
    module = ManualLayerNorm(6).double()
    x = torch.randn(2, 6, 3, dtype=torch.float64).transpose(1, 2)
    assert not x.is_contiguous()
    assert module(x).shape == x.shape
