"""Standalone deduplication verification script.

Runs MD5-based deduplication between BraTS 2021 and BraTS 2024,
generates split manifests, and logs results to MLflow.

Usage:
    python scripts/verify_deduplicate.py \
        --brats2021_dir data/raw/brats2021 \
        --brats2024_dir data/raw/brats2024 \
        --output_dir outputs/logs/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random


def set_global_seed(seed=42):
    """Set global random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed FIRST
set_global_seed(42)

import mlflow
from rich.console import Console

from data.deduplication import deduplicate_datasets
from data.splits import generate_splits
from tracking.mlflow_logger import MLflowLogger

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Deduplication and manifest generation")
    parser.add_argument("--brats2021_dir", type=str, required=True,
                        help="Path to BraTS 2021 dataset directory")
    parser.add_argument("--brats2024_dir", type=str, required=True,
                        help="Path to BraTS 2024 dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for reports and manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate paths
    brats2021_path = Path(args.brats2021_dir)
    brats2024_path = Path(args.brats2024_dir)

    if not brats2021_path.exists():
        console.print(f"[red]ERROR: BraTS 2021 directory not found: {brats2021_path}[/red]")
        sys.exit(1)
    if not brats2024_path.exists():
        console.print(f"[red]ERROR: BraTS 2024 directory not found: {brats2024_path}[/red]")
        sys.exit(1)

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri="outputs/mlruns",
        experiment_name="BrainTumorBenchmark",
    )

    with mlflow.start_run(run_name="DataPreparation"):
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("brats2021_dir", str(brats2021_path))
        mlflow.log_param("brats2024_dir", str(brats2024_path))

        # Step 1: Deduplication
        console.print("\n[bold]Step 1: Cross-year deduplication[/bold]")
        dedup_result = deduplicate_datasets(
            brats2021_dir=args.brats2021_dir,
            brats2024_dir=args.brats2024_dir,
            output_dir=args.output_dir,
        )

        # Log deduplication report as artifact
        mlflow.log_artifact(dedup_result["report_path"])
        mlflow.log_param("duplicates_found", dedup_result["report"]["duplicates_found"])
        mlflow.log_param("final_pool_size", dedup_result["report"]["final_pool_size"])

        # Step 2: Generate splits
        console.print("\n[bold]Step 2: Generating stratified splits[/bold]")
        train_df, val_df = generate_splits(
            valid_cases_2021=dedup_result["valid_cases_2021"],
            valid_cases_2024=dedup_result["valid_cases_2024"],
            output_dir=args.output_dir,
            seed=args.seed,
        )

        # Log manifests as artifacts
        train_manifest_path = Path(args.output_dir) / "train_manifest.csv"
        val_manifest_path = Path(args.output_dir) / "val_manifest.csv"
        mlflow.log_artifact(str(train_manifest_path))
        mlflow.log_artifact(str(val_manifest_path))
        mlflow.log_param("training_cases", len(train_df))
        mlflow.log_param("validation_cases", len(val_df))
        mlflow.log_param(
            "train_cases_brats2021",
            len(train_df[train_df["dataset_origin"] == "brats2021"])
        )
        mlflow.log_param(
            "train_cases_brats2024",
            len(train_df[train_df["dataset_origin"] == "brats2024"])
        )

        console.print("\n[bold green]✓ Deduplication and split generation complete![/bold green]")


if __name__ == "__main__":
    main()
