from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tasks.task05_model_pipeline import (
    ClassificationDataset,
    ResidualMLP,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def make_dataset() -> ClassificationDataset:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(48, 6, generator=generator)
    labels = (features[:, :3].sum(dim=1) > 0).long()
    return ClassificationDataset(features, labels)


def make_model() -> ResidualMLP:
    return ResidualMLP(
        input_dim=6,
        hidden_dim=16,
        num_classes=2,
        num_blocks=2,
        dropout=0.1,
    )


def test_dataset_and_model_shapes() -> None:
    dataset = make_dataset()
    assert len(dataset) == 48
    x, y = dataset[0]
    assert x.shape == (6,)
    assert y.ndim == 0

    model = make_model()
    logits = model(torch.randn(5, 6))
    assert logits.shape == (5, 2)


def test_train_evaluate_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(5)
    device = torch.device("cpu")
    loader = DataLoader(make_dataset(), batch_size=12, shuffle=False)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    train_loss = train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0)
    assert torch.isfinite(torch.tensor(train_loss))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    validation_loss, accuracy = evaluate(model, loader, device)
    assert validation_loss >= 0.0
    assert 0.0 <= accuracy <= 1.0

    path = tmp_path / "checkpoint.pt"
    model.eval()
    sample = torch.randn(4, 6)
    expected = model(sample).detach()
    save_checkpoint(path, model, optimizer, epoch=1)

    restored_model = make_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-2)
    metadata = load_checkpoint(path, restored_model, restored_optimizer)
    restored_model.eval()

    torch.testing.assert_close(restored_model(sample), expected)
    assert metadata["epoch"] == 1
