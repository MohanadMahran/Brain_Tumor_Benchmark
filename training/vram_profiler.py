"""VRAM usage logging utility for GPU memory monitoring.

Tracks peak and current memory usage throughout training and inference.
Integrates with MLflow for experiment tracking.
"""

import logging

import torch
import mlflow

logger = logging.getLogger(__name__)


class VRAMProfiler:
    """GPU VRAM usage profiler.

    Provides methods to track, log, and report GPU memory usage.
    Designed for RTX 2080 (8 GB VRAM) constraint monitoring.

    Usage:
        profiler = VRAMProfiler()
        profiler.reset()
        # ... run model ...
        peak = profiler.get_peak_mb()
        profiler.log_to_mlflow(prefix="train_")
    """

    VRAM_LIMIT_MB = 8192  # 8 GB in MB

    def __init__(self, device: int = 0):
        """Initialize profiler.

        Args:
            device: CUDA device index.
        """
        self.device = device
        self._available = torch.cuda.is_available()
        if not self._available:
            logger.warning("CUDA not available — VRAM profiling disabled.")

    def reset(self):
        """Reset peak memory statistics."""
        if self._available:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_mb(self) -> float:
        """Get peak GPU memory allocated (in MB).

        Returns:
            Peak memory in megabytes, or 0.0 if CUDA unavailable.
        """
        if not self._available:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def get_current_mb(self) -> float:
        """Get current GPU memory allocated (in MB).

        Returns:
            Current memory in megabytes, or 0.0 if CUDA unavailable.
        """
        if not self._available:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def get_reserved_mb(self) -> float:
        """Get total reserved GPU memory (in MB).

        Returns:
            Reserved memory in megabytes.
        """
        if not self._available:
            return 0.0
        return torch.cuda.memory_reserved(self.device) / (1024 ** 2)

    def check_oom_risk(self, threshold_pct: float = 0.9) -> bool:
        """Check if current usage risks OOM.

        Args:
            threshold_pct: Fraction of VRAM limit to trigger warning.

        Returns:
            True if peak usage exceeds threshold.
        """
        peak = self.get_peak_mb()
        if peak > self.VRAM_LIMIT_MB * threshold_pct:
            logger.warning(
                f"VRAM usage high: {peak:.0f} MB / {self.VRAM_LIMIT_MB} MB "
                f"({peak / self.VRAM_LIMIT_MB * 100:.1f}%)"
            )
            return True
        return False

    def log_to_mlflow(self, prefix: str = "", step: int = None):
        """Log current VRAM stats to MLflow.

        Args:
            prefix: Metric name prefix (e.g., "train_", "val_").
            step: Step number for metric logging.
        """
        peak = self.get_peak_mb()
        current = self.get_current_mb()
        try:
            if step is not None:
                mlflow.log_metric(f"{prefix}vram_peak_mb", peak, step=step)
                mlflow.log_metric(f"{prefix}vram_current_mb", current, step=step)
            else:
                mlflow.log_metric(f"{prefix}vram_peak_mb", peak)
                mlflow.log_metric(f"{prefix}vram_current_mb", current)
        except Exception as e:
            logger.warning(f"Failed to log VRAM to MLflow: {e}")

    def get_summary(self) -> dict:
        """Get summary of current VRAM state.

        Returns:
            Dict with peak_mb, current_mb, reserved_mb, utilization_pct.
        """
        peak = self.get_peak_mb()
        return {
            "peak_mb": peak,
            "current_mb": self.get_current_mb(),
            "reserved_mb": self.get_reserved_mb(),
            "utilization_pct": peak / self.VRAM_LIMIT_MB * 100,
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"VRAMProfiler(peak={summary['peak_mb']:.0f}MB, "
            f"current={summary['current_mb']:.0f}MB, "
            f"utilization={summary['utilization_pct']:.1f}%)"
        )
