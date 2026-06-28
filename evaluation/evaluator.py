"""Benchmark evaluator using sliding window inference.

Evaluates trained models on held-out benchmark datasets (TCGA, BraTS-SSA)
using MONAI's sliding_window_inference with Gaussian blending.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from rich.console import Console
from rich.progress import Progress
from torch.cuda.amp import autocast

from data.dataset import BenchmarkDataset
from training.metrics import compute_all_metrics
from training.vram_profiler import VRAMProfiler

console = Console()
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluator for benchmark datasets using sliding window inference.

    Uses MONAI sliding_window_inference with identical parameters for both models.
    Reports per-case and aggregate metrics.

    Args:
        model: Trained model.
        model_name: Name for logging.
        device: CUDA device.
        eval_patch_size: Sliding window ROI size.
        overlap: Overlap fraction between windows.
        sw_batch_size: Number of windows processed at once.
        use_amp: Whether to use mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device = None,
        eval_patch_size: tuple = (128, 128, 128),
        overlap: float = 0.25,
        sw_batch_size: int = 1,
        use_amp: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_patch_size = eval_patch_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.use_amp = use_amp
        self.model = self.model.to(self.device)
        self.model.eval()
        self.vram_profiler = VRAMProfiler()

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        output_dir: str,
        mlflow_run_name: str,
        val_dice_mean: float = 0.0,
    ) -> Dict:
        """Evaluate model on an entire benchmark dataset.

        Args:
            dataset: BenchmarkDataset instance.
            output_dir: Directory to save per-case results.
            mlflow_run_name: Name for the MLflow run.
            val_dice_mean: Validation dice mean for generalization gap calculation.

        Returns:
            Aggregate metrics dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        case_results = []
        self.vram_profiler.reset()

        with mlflow.start_run(run_name=mlflow_run_name):
            # Log parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("benchmark_dataset", dataset.benchmark_type)
            mlflow.log_param("num_benchmark_cases", len(dataset))
            mlflow.log_param("eval_patch_size", str(self.eval_patch_size))
            mlflow.log_param("sliding_window_overlap", self.overlap)
            mlflow.log_param("sw_batch_size", self.sw_batch_size)
            mlflow.log_param("mixed_precision", self.use_amp)

            with Progress() as progress:
                task = progress.add_task(
                    f"Evaluating {self.model_name} on {dataset.benchmark_type}...",
                    total=len(dataset),
                )

                for idx in range(len(dataset)):
                    try:
                        result = self._evaluate_single_case(dataset, idx)
                        case_results.append(result)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            logger.warning(
                                f"OOM on case {idx}, skipping. "
                                f"Peak VRAM: {self.vram_profiler.get_peak_mb():.0f} MB"
                            )
                            mlflow.log_param("oom_cases", True)
                        else:
                            raise
                    progress.advance(task)

            # Aggregate metrics
            aggregate = self._aggregate_results(case_results, val_dice_mean)

            # Log aggregate metrics to MLflow
            for key, value in aggregate.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # VRAM logging
            self.vram_profiler.log_to_mlflow(prefix="benchmark_")

            # Save per-case results
            results_df = pd.DataFrame(case_results)
            results_path = output_path / f"{self.model_name}_results.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))

            logger.info(
                f"Benchmark complete: {self.model_name} on {dataset.benchmark_type}. "
                f"Dice mean: {aggregate.get('dice_mean', 0.0):.4f}"
            )

        return aggregate

    def _evaluate_single_case(self, dataset: BenchmarkDataset, idx: int) -> Dict:
        """Evaluate a single case using sliding window inference.

        Args:
            dataset: Benchmark dataset.
            idx: Case index.

        Returns:
            Dict with case metrics and metadata.
        """
        start_time = time.time()
        data = dataset[idx]
        image = data["image"].unsqueeze(0).to(self.device)  # (1, 4, D, H, W)
        label = data["label"].unsqueeze(0) if "label" in data else None

        # Sliding window inference
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    predictor = lambda x: torch.sigmoid(self.model(x))
                    outputs = sliding_window_inference(
                        inputs=image,
                        roi_size=self.eval_patch_size,
                        sw_batch_size=self.sw_batch_size,
                        predictor=predictor,
                        overlap=self.overlap,
                        mode="gaussian",
                        progress=False,
                    )
            else:
                predictor = lambda x: torch.sigmoid(self.model(x))
                outputs = sliding_window_inference(
                    inputs=image,
                    roi_size=self.eval_patch_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=predictor,
                    overlap=self.overlap,
                    mode="gaussian",
                    progress=False,
                )

        inference_time = time.time() - start_time

        # Threshold and compute metrics
        pred_binary = (outputs > 0.5).cpu().numpy()[0]  # (3, D, H, W)

        result = {
            "case_id": data["case_id"],
            "tumor_type": data.get("tumor_type", "unknown"),
            "inference_time_s": inference_time,
        }

        if label is not None:
            target_binary = label.numpy()[0]
            metrics = compute_all_metrics(pred_binary, target_binary)
            result.update(metrics)

        return result

    def _aggregate_results(
        self,
        case_results: List[Dict],
        val_dice_mean: float,
    ) -> Dict:
        """Aggregate per-case results into summary metrics.

        Args:
            case_results: List of per-case metric dicts.
            val_dice_mean: Validation dice for generalization gap.

        Returns:
            Aggregate metrics dictionary.
        """
        if not case_results:
            return {"dice_mean": 0.0, "num_cases": 0}

        # Filter results that have metrics (label was available)
        with_metrics = [r for r in case_results if "dice_mean" in r]
        if not with_metrics:
            return {"dice_mean": 0.0, "num_cases": len(case_results)}

        aggregate = {"num_benchmark_cases": len(with_metrics)}

        # Overall metrics
        metric_keys = ["dice_ET", "dice_TC", "dice_WT", "dice_mean",
                       "hd95_ET", "hd95_TC", "hd95_WT", "hd95_mean"]
        for key in metric_keys:
            values = [r[key] for r in with_metrics if key in r]
            if values:
                aggregate[key] = float(np.mean(values))

        # Generalization gap
        benchmark_dice_mean = aggregate.get("dice_mean", 0.0)
        aggregate["dice_drop_vs_val"] = val_dice_mean - benchmark_dice_mean

        # Per-tumor-type breakdown (for TCGA)
        tumor_types = set(r.get("tumor_type", "unknown") for r in with_metrics)
        for tt in tumor_types:
            if tt in ("GBM", "LGG"):
                tt_results = [r for r in with_metrics if r.get("tumor_type") == tt]
                if tt_results:
                    tt_dice = [r["dice_mean"] for r in tt_results]
                    aggregate[f"dice_mean_{tt}"] = float(np.mean(tt_dice))

        # Inference time
        times = [r["inference_time_s"] for r in case_results]
        aggregate["mean_inference_time_s"] = float(np.mean(times))

        # VRAM
        aggregate["vram_peak_mb"] = self.vram_profiler.get_peak_mb()

        return aggregate
