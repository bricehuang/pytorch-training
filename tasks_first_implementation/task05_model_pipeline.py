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
        self.ln = nn.LayerNorm(normalized_shape=hidden_dim)
        self.l1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.l2(self.dropout(self.gelu(self.l1(self.ln(x)))))


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
        self.lin = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.lout = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        for block in self.blocks:
            h = block(h)
        return self.lout(h)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    model = model.to(device)
    num_examples = 0
    sum_loss = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        n = len(labels)
        num_examples += n
        sum_loss += n * loss.item()
    return sum_loss / num_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` for a classification loader."""
    model.eval()
    model = model.to(device)
    num_examples = 0
    num_correct = 0
    sum_loss = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits: torch.Tensor = model(features)
        loss = float(F.cross_entropy(logits, labels))
        predictions = logits.argmax(dim=-1)
        correct = int((predictions == labels).sum())
        n = len(labels)
        num_examples += n
        num_correct += correct
        sum_loss += n * loss
    return (sum_loss / num_examples, num_correct / num_examples)


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
