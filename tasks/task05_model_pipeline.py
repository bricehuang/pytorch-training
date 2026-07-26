"""Task 5: Dataset, DataLoader, nn.Module, training, evaluation, checkpoints."""

from __future__ import annotations

from os import PathLike
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A map-style tensor classification dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class ResidualBlock(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual add."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(self.dropout(self.gelu(self.lin1(self.ln(x)))))


class ResidualMLP(nn.Module):
    """Residual MLP classifier for input tensors with shape ``[B, D]``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.lin_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        h = self.lin_in(x)
        for block in self.blocks:
            h = block(h)
        return self.lin_out(h)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch and return mean loss per example."""
    model.to(device)
    model.train()
    num_examples = 0
    total_loss = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        n = len(labels)
        num_examples += n
        total_loss += n * loss.item()
    return total_loss / num_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` for a classification loader."""
    model.to(device)
    model.eval()
    num_examples = 0
    num_correct = 0
    total_loss = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        n = len(labels)
        c = int((predictions == labels).sum())
        num_examples += n
        num_correct += c
        total_loss += n * loss.item()
    return (total_loss / num_examples, num_correct / num_examples)


def save_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **metadata: Any,
) -> None:
    """Save model state, optimizer state, and optional metadata."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": metadata
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optimizer, returning saved metadata."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["metadata"]


def summarize_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[int, int, torch.Tensor]:
    dataset = ClassificationDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_examples = 0
    total_positive = 0
    sum_feature = torch.zeros_like(features[0])
    for f, l in loader:
        total_examples += len(l)
        total_positive += int((l == 1).sum())
        sum_feature += f.sum(dim=0, keepdim=False)
    mean_feature = sum_feature / total_examples
    assert torch.allclose(mean_feature, features.mean(dim=0))
    return (total_examples, total_positive, mean_feature)
