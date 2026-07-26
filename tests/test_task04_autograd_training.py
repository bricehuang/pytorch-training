import torch

from tasks.task04_autograd_training import train_logistic_regression


def make_linearly_separable_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    X = torch.randn(160, 3, generator=generator)
    true_w = torch.tensor([1.8, -2.2, 1.1])
    y = ((X @ true_w + 0.2) > 0).float()
    return X, y


def test_manual_logistic_training_learns_separable_data() -> None:
    X, y = make_linearly_separable_data()
    weight, bias, history = train_logistic_regression(
        X,
        y,
        learning_rate=0.2,
        steps=250,
    )

    assert weight.shape == (X.shape[1], 1)
    assert bias.shape == (1,)
    assert len(history) == 250
    assert history[-1] < history[0]

    logits = (X @ weight).squeeze(-1) + bias
    predictions = (logits > 0).float()
    accuracy = (predictions == y).float().mean().item()
    assert accuracy > 0.95


def test_returned_parameters_are_finite_leaf_tensors() -> None:
    X, y = make_linearly_separable_data()
    weight, bias, _ = train_logistic_regression(
        X,
        y,
        learning_rate=0.1,
        steps=5,
    )
    assert weight.is_leaf
    assert bias.is_leaf
    assert torch.isfinite(weight).all()
    assert torch.isfinite(bias).all()
