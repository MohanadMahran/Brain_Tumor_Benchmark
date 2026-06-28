"""MLflow logging wrapper for brain tumor segmentation experiments.

Provides a clean interface for:
    - Experiment creation and run management
    - Parameter and metric logging
    - Artifact management
    - Graceful run completion (no orphaned RUNNING states)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class MLflowLogger:
    """MLflow experiment tracking wrapper.

    Ensures all runs end as FINISHED or FAILED — never orphaned RUNNING.
    Manages experiment creation and artifact logging.

    Args:
        tracking_uri: Path to MLflow tracking store.
        experiment_name: Name of the experiment.
    """

    def __init__(
        self,
        tracking_uri: str = "outputs/mlruns",
        experiment_name: str = "BrainTumorBenchmark",
    ):
        self.tracking_uri = str(Path(tracking_uri).resolve())
        self.experiment_name = experiment_name
        self._active_run = None

        # Set tracking URI
        mlflow.set_tracking_uri(f"file://{self.tracking_uri}")

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        logger.info(
            f"MLflow initialized: uri={self.tracking_uri}, "
            f"experiment={experiment_name}"
        )

    def start_run(self, run_name: str, params: Dict[str, Any] = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Descriptive run name.
            params: Initial parameters to log.

        Returns:
            Run ID string.
        """
        self._active_run = mlflow.start_run(run_name=run_name)
        run_id = self._active_run.info.run_id

        if params:
            self.log_params(params)

        logger.info(f"MLflow run started: {run_name} ({run_id})")
        return run_id

    def end_run(self, status: str = "FINISHED"):
        """End the active run with specified status.

        Args:
            status: Run status — 'FINISHED' or 'FAILED'.
        """
        if self._active_run is not None:
            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended: status={status}")
            self._active_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters.

        Args:
            params: Dict of parameter names and values.
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics.

        Args:
            metrics: Dict of metric names and values.
            step: Step number.
        """
        for key, value in metrics.items():
            try:
                if step is not None:
                    mlflow.log_metric(key, value, step=step)
                else:
                    mlflow.log_metric(key, value)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_artifact(self, path: str):
        """Log a file as an artifact.

        Args:
            path: Path to the file.
        """
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_hardware_context(self):
        """Log GPU and CUDA hardware context."""
        import torch

        params = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cuda_version": torch.version.cuda or "N/A",
            "pytorch_version": torch.__version__,
            "gpu_memory_gb": (
                torch.cuda.get_device_properties(0).total_mem / (1024**3)
                if torch.cuda.is_available() else 0
            ),
        }
        self.log_params(params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
        return False
