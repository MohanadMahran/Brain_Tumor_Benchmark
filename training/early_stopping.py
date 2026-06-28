"""Early stopping with patience for validation metric monitoring.

Monitors validation mean Dice score and stops training when no improvement
is observed for a specified number of epochs.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric (higher is better) and triggers when no improvement
    is seen for `patience` consecutive epochs.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement to qualify as improvement.
        mode: 'max' (metric should increase) or 'min' (metric should decrease).
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.triggered = False
        self.triggered_at_epoch = None
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        improved = self._is_improvement(value)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                self.triggered_at_epoch = epoch
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best value: {self.best_value:.4f} at epoch {self.best_epoch}. "
                    f"No improvement for {self.patience} epochs."
                )
                return True

        return False

    def _is_improvement(self, value: float) -> bool:
        """Check if value represents an improvement over best.

        Args:
            value: Current metric value.

        Returns:
            True if value is better than best by at least min_delta.
        """
        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        else:
            return value < (self.best_value - self.min_delta)

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "triggered": self.triggered,
            "triggered_at_epoch": self.triggered_at_epoch,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.best_value = state["best_value"]
        self.counter = state["counter"]
        self.triggered = state["triggered"]
        self.triggered_at_epoch = state["triggered_at_epoch"]
        self.best_epoch = state["best_epoch"]
