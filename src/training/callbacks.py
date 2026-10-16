"""
src/training/callbacks.py

Training callbacks for PyTorch Autoencoders.

Protocol Rule:
  Validation data (`val`) is strictly reserved for anomaly threshold calibration
  after training completes. Early stopping and LR scheduling monitor ONLY the
  healthy training loss convergence (`train_loss`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class TrainingHistory:
    """Stores per-epoch training metrics."""

    train_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = float("inf")
    stopped_early: bool = False


class TrainConvergenceEarlyStopping:
    """
    Monitors healthy training loss convergence and restores the best weights.
    Never accesses validation or test splits.
    """

    def __init__(self, patience: int = 30, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state_dict: dict[str, Any] | None = None

    def step(self, current_loss: float, epoch: int, model: nn.Module) -> bool:
        """
        Update state with current epoch's training loss.
        Returns True if training has converged and should stop early.
        """
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the model parameters from the best training epoch."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

