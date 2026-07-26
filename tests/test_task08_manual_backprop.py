import torch
import torch.nn.functional as F

from tasks.task08_manual_backprop import manual_mlp_loss_and_gradients


def make_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(9)
    X = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    y = torch.tensor([0, 2, 1, 2, 0])
    W1 = torch.randn(4, 6, generator=generator, dtype=torch.float64) / 3
    b1 = torch.randn(6, generator=generator, dtype=torch.float64) / 3
    W2 = torch.randn(6, 3, generator=generator, dtype=torch.float64) / 3
    b2 = torch.randn(3, generator=generator, dtype=torch.float64) / 3
    return X, y, W1, b1, W2, b2


def test_manual_gradients_match_autograd() -> None:
    X, y, W1, b1, W2, b2 = make_inputs()
    manual_loss, manual_grads = manual_mlp_loss_and_gradients(X, y, W1, b1, W2, b2)

    references = [tensor.detach().clone().requires_grad_(True) for tensor in (W1, b1, W2, b2)]
    rW1, rb1, rW2, rb2 = references
    logits = torch.tanh(X @ rW1 + rb1) @ rW2 + rb2
    reference_loss = F.cross_entropy(logits, y)
    reference_loss.backward()

    torch.testing.assert_close(manual_loss, reference_loss.detach())
    for name, reference in zip(("W1", "b1", "W2", "b2"), references):
        torch.testing.assert_close(manual_grads[name], reference.grad, rtol=1e-6, atol=1e-8)


def test_directional_finite_difference() -> None:
    X, y, W1, b1, W2, b2 = make_inputs()
    loss, grads = manual_mlp_loss_and_gradients(X, y, W1, b1, W2, b2)
    del loss

    generator = torch.Generator().manual_seed(10)
    directions = {
        "W1": torch.randn(W1.shape, generator=generator, dtype=W1.dtype),
        "b1": torch.randn(b1.shape, generator=generator, dtype=b1.dtype),
        "W2": torch.randn(W2.shape, generator=generator, dtype=W2.dtype),
        "b2": torch.randn(b2.shape, generator=generator, dtype=b2.dtype),
    }
    epsilon = 1e-6

    plus = manual_mlp_loss_and_gradients(
        X,
        y,
        W1 + epsilon * directions["W1"],
        b1 + epsilon * directions["b1"],
        W2 + epsilon * directions["W2"],
        b2 + epsilon * directions["b2"],
    )[0]
    minus = manual_mlp_loss_and_gradients(
        X,
        y,
        W1 - epsilon * directions["W1"],
        b1 - epsilon * directions["b1"],
        W2 - epsilon * directions["W2"],
        b2 - epsilon * directions["b2"],
    )[0]

    finite_difference = (plus - minus) / (2 * epsilon)
    directional_derivative = sum((grads[name] * directions[name]).sum() for name in grads)
    torch.testing.assert_close(finite_difference, directional_derivative, rtol=1e-5, atol=1e-7)
