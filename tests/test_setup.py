import torch


def test_cpu_matmul_and_autograd() -> None:
    """Verify that basic CPU operations and autograd work."""

    x = torch.randn(8, 16, requires_grad=True)

    scores = x @ x.transpose(0, 1)
    loss = scores.square().mean()
    loss.backward()

    assert scores.shape == (8, 8)
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.cuda.is_available()
