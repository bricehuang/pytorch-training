import torch
import torch.nn.functional as F

from tasks.task03_stable_losses import (
    cross_entropy_from_logits,
    masked_softmax,
    stable_log_softmax,
)


def test_stable_log_softmax_handles_large_logits() -> None:
    logits = torch.tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        dtype=torch.float64,
    )
    actual = stable_log_softmax(logits, dim=-1)
    expected = F.log_softmax(logits, dim=-1)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_masked_softmax() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    probabilities = masked_softmax(scores, mask)

    assert probabilities.shape == scores.shape
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))

    expected = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    torch.testing.assert_close(probabilities, expected)


def test_cross_entropy_output_and_gradient() -> None:
    torch.manual_seed(1)
    targets = torch.tensor([0, 3, 1, 2])

    logits_actual = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    logits_expected = logits_actual.detach().clone().requires_grad_(True)

    actual = cross_entropy_from_logits(logits_actual, targets)
    expected = F.cross_entropy(logits_expected, targets)
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits_actual.grad, logits_expected.grad)
