"""Task 5: Dataset, DataLoader, nn.Module, training, evaluation, checkpoints."""

from __future__ import annotations

from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A map-style tensor classification dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        raise NotImplementedError("Implement ClassificationDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("Implement ClassificationDataset.__len__")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement ClassificationDataset.__getitem__")


class ResidualBlock(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual add."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        raise NotImplementedError("Implement ResidualBlock.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement ResidualBlock.forward")


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
        raise NotImplementedError("Implement ResidualMLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes]``."""
        raise NotImplementedError("Implement ResidualMLP.forward")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch and return mean loss per example."""
    raise NotImplementedError("Implement train_one_epoch")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` for a classification loader."""
    raise NotImplementedError("Implement evaluate")


def save_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **metadata: Any,
) -> None:
    """Save model state, optimizer state, and optional metadata."""
    raise NotImplementedError("Implement save_checkpoint")


def load_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optimizer, returning saved metadata."""
    raise NotImplementedError("Implement load_checkpoint")
