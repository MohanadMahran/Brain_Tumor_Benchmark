"""Entry point: Run benchmark evaluation for all models.

Evaluates U-Net, and UNETR on held-out benchmark datasets using
sliding window inference with identical parameters.

Usage:
    python scripts/benchmark.py \\
        --benchmark upenn_gbm \\
        --unet_checkpoint outputs/models/unet3d/best_checkpoint.pth \\
        --unetr_checkpoint outputs/models/unetr/best_checkpoint.pth \\
        --data_dir data/raw/UPenn-GBM \\
        --output_dir outputs/reports/upenn_gbm
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
        model_type: 'unet3d', 'unetr'.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    parser.add_argument("--benchmark", type=str, required=True, choices=["upenn_gbm", "ssa"])
    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--unetr_checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--case_id", type=str, default=None,
        help="Evaluate only a single case by its ID (e.g. UPENN-GBM-00307). "
             "If not set, evaluates the full benchmark dataset.",
    )
    args = parser.parse_args()

    # Load base config for evaluation parameters
    if Path(args.base_config).exists():
        with open(args.base_config, "r") as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {}

    eval_patch_size = tuple(base_config.get("eval_patch_size", [128, 128, 128]))
    overlap = base_config.get("sliding_window_overlap", 0.25)

    # Validate paths and build list of models to evaluate
    all_ckpts = {
        "unet3d": ("UNet3D", args.unet_checkpoint),
        "unetr": ("UNETR", args.unetr_checkpoint),
    }
    
    model_ckpts = {}
    for model_type, (display_name, ckpt_path) in all_ckpts.items():
        if ckpt_path is not None:
            if not Path(ckpt_path).exists():
                console.print(f"[red]ERROR: {display_name} checkpoint not found: {ckpt_path}[/red]")
                sys.exit(1)
            model_ckpts[model_type] = (display_name, ckpt_path)

    if not model_ckpts:
        console.print("[red]ERROR: At least one model checkpoint must be provided for evaluation.[/red]")
        sys.exit(1)

    if not Path(args.data_dir).exists():
        console.print(f"[red]ERROR: Benchmark data directory not found: {args.data_dir}[/red]")
        sys.exit(1)

    # Load models
    console.print(f"\n[bold]Loading models for {args.benchmark.upper()}...[/bold]")
    models = {}
    val_dices = {}
    for model_type, (display_name, ckpt_path) in model_ckpts.items():
        model, val_dice = load_model_from_checkpoint(ckpt_path, model_type)
        models[model_type] = model
        val_dices[model_type] = val_dice

    # Load benchmark dataset
    console.print(f"\n[bold]Loading {args.benchmark.upper()} benchmark dataset...[/bold]")
    dataset = BenchmarkDataset(
        data_dir=args.data_dir,
        benchmark_type=args.benchmark,
    )
    console.print(f"  Found {len(dataset)} cases")

    # Filter to single case if --case_id is specified
    if args.case_id:
        original_count = len(dataset.cases)
        dataset.cases = [
            c for c in dataset.cases if c["case_id"] == args.case_id
        ]
        if not dataset.cases:
            console.print(
                f"[red]ERROR: Case '{args.case_id}' not found in dataset. "
                f"Available cases ({original_count}): "
                f"{', '.join(c['case_id'] for c in BenchmarkDataset(args.data_dir, args.benchmark).cases[:5])}...[/red]"
            )
            sys.exit(1)
        console.print(f"  Filtered to single case: {args.case_id}")

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri=base_config.get("mlflow_tracking_uri", "outputs/mlruns"),
        experiment_name=base_config.get("experiment_name", "BrainTumorBenchmark"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_name = "UPennGBM" if args.benchmark == "upenn_gbm" else "SSA"

    # Evaluate each model
    all_results = {}
    for model_type, (display_name, _) in model_ckpts.items():
        console.print(f"\n[bold]Evaluating {display_name} on {args.benchmark.upper()}...[/bold]")
        evaluator = BenchmarkEvaluator(
            model=models[model_type],
            model_name=model_type,
            device=device,
            eval_patch_size=eval_patch_size,
            overlap=overlap,
            sw_batch_size=1,
            use_amp=True,
        )
        output_dir = str(Path(args.output_dir) / "single_case") if args.case_id else args.output_dir
        results = evaluator.evaluate_dataset(
            dataset=dataset,
            output_dir=output_dir,
            mlflow_run_name=f"BrainTumor_{display_name}_Benchmark_{benchmark_name}",
            val_dice_mean=val_dices[model_type],
        )
        all_results[model_type] = results

        # Free VRAM between models
        del models[model_type], evaluator
        torch.cuda.empty_cache()

    # Print comparison
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]{args.benchmark.upper()} Benchmark Results[/bold]")
    console.print(f"[bold]{'='*60}[/bold]")
    for model_type, (display_name, _) in model_ckpts.items():
        if model_type in all_results:
            r = all_results[model_type]
            console.print(
                f"  {display_name:12s} Dice Mean: {r.get('dice_mean', 0.0):.4f}  "
                f"HD95 Mean: {r.get('hd95_mean', 0.0):.2f}"
            )

    if args.benchmark == "upenn_gbm":
        console.print(f"\n  Per tumor type:")
        for model_type, (display_name, _) in model_ckpts.items():
            if model_type in all_results:
                r = all_results[model_type]
                console.print(
                    f"    {display_name:12s} GBM: {r.get('dice_mean_GBM', 0.0):.4f}  "
                    f"LGG: {r.get('dice_mean_LGG', 0.0):.4f}"
                )


if __name__ == "__main__":
    main()

