"""Entry point: Run benchmark evaluation for both models.

Evaluates U-Net and UNETR on held-out benchmark datasets using
sliding window inference with identical parameters.

Usage:
    python scripts/benchmark.py \
        --benchmark tcga \
        --unet_checkpoint outputs/models/unet3d/best_checkpoint.pth \
        --unetr_checkpoint outputs/models/unetr/best_checkpoint.pth \
        --data_dir data/raw/tcga \
        --output_dir outputs/reports/tcga
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

import yaml
from rich.console import Console

from models.unet3d import UNet3D
from models.unetr import UNETR
from models.param_counter import count_parameters, verify_parameter_parity
from data.dataset import BenchmarkDataset
from evaluation.evaluator import BenchmarkEvaluator
from tracking.mlflow_logger import MLflowLogger

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, model_type: str) -> tuple:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model_type: 'unet3d' or 'unetr'.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    if model_type == "unet3d":
        model = UNet3D(
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 3),
            base_channels=config.get("base_channels", 32),
            groups=config.get("groups", 8),
            dropout_rates=config.get("dropout_rates", [0.1, 0.2, 0.2, 0.2]),
            drop_path_rate=config.get("drop_path_rate", 0.1),
        )
    elif model_type == "unetr":
        model = UNETR(
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 3),
            input_size=config.get("patch_size", [96, 96, 96])[0] if isinstance(config.get("patch_size", [96, 96, 96]), (list, tuple)) else config.get("patch_size", 96),
            patch_size=config.get("patch_size_tokens", 16),
            embedding_dim=config.get("embedding_dim", 256),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 4),
            mlp_ratio=config.get("mlp_ratio", 4),
            dropout=config.get("dropout", 0.1),
            drop_path_rate=config.get("drop_path_rate", 0.1),
            use_checkpoint=False,  # No checkpointing during inference
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = count_parameters(model)
    val_dice = checkpoint.get("val_dice_mean", 0.0)

    console.print(
        f"  Loaded {model_type}: epoch={checkpoint.get('epoch', '?')}, "
        f"params={params:,}, val_dice={val_dice:.4f}"
    )
    return model, val_dice


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--benchmark", type=str, required=True, choices=["tcga", "ssa"])
    parser.add_argument("--unet_checkpoint", type=str, required=True)
    parser.add_argument("--unetr_checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    # Load base config for evaluation parameters
    if Path(args.base_config).exists():
        with open(args.base_config, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {}

    eval_patch_size = tuple(base_config.get("eval_patch_size", [128, 128, 128]))
    overlap = base_config.get("sliding_window_overlap", 0.25)

    # Validate paths
    if not Path(args.unet_checkpoint).exists():
        console.print(f"[red]ERROR: U-Net checkpoint not found: {args.unet_checkpoint}[/red]")
        sys.exit(1)
    if not Path(args.unetr_checkpoint).exists():
        console.print(f"[red]ERROR: UNETR checkpoint not found: {args.unetr_checkpoint}[/red]")
        sys.exit(1)
    if not Path(args.data_dir).exists():
        console.print(f"[red]ERROR: Benchmark data directory not found: {args.data_dir}[/red]")
        sys.exit(1)

    # Load models
    console.print(f"\n[bold]Loading models for {args.benchmark.upper()}...[/bold]")
    unet_model, unet_val_dice = load_model_from_checkpoint(args.unet_checkpoint, "unet3d")
    unetr_model, unetr_val_dice = load_model_from_checkpoint(args.unetr_checkpoint, "unetr")

    # Verify parameter parity
    verify_parameter_parity(unet_model, unetr_model, "UNet3D", "UNETR", tolerance=0.05)

    # Load benchmark dataset
    console.print(f"\n[bold]Loading {args.benchmark.upper()} benchmark dataset...[/bold]")
    dataset = BenchmarkDataset(
        data_dir=args.data_dir,
        benchmark_type=args.benchmark,
    )
    console.print(f"  Found {len(dataset)} cases")

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri=base_config.get("mlflow_tracking_uri", "outputs/mlruns"),
        experiment_name=base_config.get("experiment_name", "BrainTumorBenchmark"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate U-Net
    console.print(f"\n[bold]Evaluating UNet3D on {args.benchmark.upper()}...[/bold]")
    unet_evaluator = BenchmarkEvaluator(
        model=unet_model,
        model_name="unet3d",
        device=device,
        eval_patch_size=eval_patch_size,
        overlap=overlap,
        sw_batch_size=1,
        use_amp=True,
    )

    benchmark_name = "TCGA" if args.benchmark == "tcga" else "SSA"
    unet_results = unet_evaluator.evaluate_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        mlflow_run_name=f"BrainTumor_UNet3D_Benchmark_{benchmark_name}",
        val_dice_mean=unet_val_dice,
    )

    # Free VRAM
    del unet_model, unet_evaluator
    torch.cuda.empty_cache()

    # Evaluate UNETR
    console.print(f"\n[bold]Evaluating UNETR on {args.benchmark.upper()}...[/bold]")
    unetr_evaluator = BenchmarkEvaluator(
        model=unetr_model,
        model_name="unetr",
        device=device,
        eval_patch_size=eval_patch_size,
        overlap=overlap,
        sw_batch_size=1,
        use_amp=True,
    )

    unetr_results = unetr_evaluator.evaluate_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        mlflow_run_name=f"BrainTumor_UNETR_Benchmark_{benchmark_name}",
        val_dice_mean=unetr_val_dice,
    )

    # Print comparison
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]{args.benchmark.upper()} Benchmark Results[/bold]")
    console.print(f"[bold]{'='*60}[/bold]")
    console.print(f"  UNet3D Dice Mean: {unet_results.get('dice_mean', 0.0):.4f}")
    console.print(f"  UNETR  Dice Mean: {unetr_results.get('dice_mean', 0.0):.4f}")
    console.print(f"  UNet3D HD95 Mean: {unet_results.get('hd95_mean', 0.0):.2f}")
    console.print(f"  UNETR  HD95 Mean: {unetr_results.get('hd95_mean', 0.0):.2f}")

    if args.benchmark == "tcga":
        console.print(f"\n  Per tumor type:")
        console.print(f"    UNet3D GBM: {unet_results.get('dice_mean_GBM', 0.0):.4f}")
        console.print(f"    UNet3D LGG: {unet_results.get('dice_mean_LGG', 0.0):.4f}")
        console.print(f"    UNETR  GBM: {unetr_results.get('dice_mean_GBM', 0.0):.4f}")
        console.print(f"    UNETR  LGG: {unetr_results.get('dice_mean_LGG', 0.0):.4f}")


if __name__ == "__main__":
    main()
