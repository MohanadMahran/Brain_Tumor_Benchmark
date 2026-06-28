"""Entry point: Train 3D U-Net for brain tumor segmentation.

Usage:
    python scripts/train_unet3d.py \
        --config configs/unet3d.yaml \
        --base_config configs/base.yaml \
        --output_dir outputs/models/unet3d \
        --log_dir outputs/logs/unet3d
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


# Set seed FIRST — before any other imports that might use randomness
set_global_seed(42)

import mlflow
import yaml
from torch.utils.data import DataLoader
from rich.console import Console

from models.unet3d import UNet3D
from models.param_counter import count_parameters, print_model_summary
from data.dataset import BraTSDataset
from data.augmentation import get_train_transforms, get_val_transforms
from training.trainer import Trainer
from training.vram_profiler import VRAMProfiler
from tracking.mlflow_logger import MLflowLogger

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str, base_config_path: str) -> dict:
    """Load and merge model config with base config.

    Args:
        config_path: Path to model-specific config.
        base_config_path: Path to shared base config.

    Returns:
        Merged configuration dictionary.
    """
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Merge: model config overrides base
    config = {**base_config, **model_config}
    return config


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, default="outputs/logs")
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)

    # Allow TF32 for Turing GPUs
    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    console.print("\n[bold]Building 3D U-Net...[/bold]")
    model = UNet3D(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 3),
        base_channels=config.get("base_channels", 32),
        groups=config.get("groups", 8),
        dropout_rates=config.get("dropout_rates", [0.1, 0.2, 0.2, 0.2]),
        drop_path_rate=config.get("drop_path_rate", 0.1),
    )

    print_model_summary(model, "UNet3D")
    total_params = count_parameters(model)
    console.print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Verify no BatchNorm
    for m in model.modules():
        assert not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)), \
            "BatchNorm detected in UNet3D! This is not allowed."

    # Build datasets and dataloaders
    console.print("\n[bold]Loading datasets...[/bold]")
    train_manifest = Path(args.manifest_dir) / "train_manifest.csv"
    val_manifest = Path(args.manifest_dir) / "val_manifest.csv"

    if not train_manifest.exists():
        console.print(f"[red]ERROR: Train manifest not found: {train_manifest}[/red]")
        console.print("[yellow]Run deduplication first: python scripts/verify_deduplicate.py[/yellow]")
        sys.exit(1)

    train_transforms = get_train_transforms(
        patch_size=tuple(config.get("patch_size", [96, 96, 96]))
    )
    val_transforms = get_val_transforms(
        patch_size=tuple(config.get("eval_patch_size", [128, 128, 128]))
    )

    train_dataset = BraTSDataset(
        manifest_path=str(train_manifest),
        transform=train_transforms,
    )
    val_dataset = BraTSDataset(
        manifest_path=str(val_manifest),
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("physical_batch_size", 1),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
    )

    console.print(f"  Train: {len(train_dataset)} cases")
    console.print(f"  Val: {len(val_dataset)} cases")

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri=config.get("mlflow_tracking_uri", "outputs/mlruns"),
        experiment_name=config.get("experiment_name", "BrainTumorBenchmark"),
    )

    # Start MLflow run
    with mlflow.start_run(run_name="BrainTumor_UNet3D_Training"):
        # Log parameters
        mlflow.log_param("model_name", "unet3d")
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("effective_batch_size", config.get("effective_batch_size", 2))
        mlflow.log_param("physical_batch_size", config.get("physical_batch_size", 1))
        mlflow.log_param("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 2))
        mlflow.log_param("patch_size_train", str(config.get("patch_size", [96, 96, 96])))
        mlflow.log_param("patch_size_eval", str(config.get("eval_patch_size", [128, 128, 128])))
        mlflow.log_param("sliding_window_overlap", config.get("sliding_window_overlap", 0.25))
        mlflow.log_param("optimizer", config.get("optimizer", "adamw"))
        mlflow.log_param("learning_rate", config.get("learning_rate", 3e-4))
        mlflow.log_param("weight_decay", config.get("weight_decay", 1e-5))
        mlflow.log_param("warmup_epochs", config.get("warmup_epochs", 10))
        mlflow.log_param("loss_function", "DiceCE")
        mlflow.log_param("normalization", "GroupNorm")
        mlflow.log_param("mixed_precision", config.get("mixed_precision", True))
        mlflow.log_param("gradient_checkpointing", False)
        mlflow.log_param("training_cases", len(train_dataset))
        mlflow.log_param("validation_cases", len(val_dataset))
        mlflow.log_param("seed", config.get("seed", 42))
        mlflow.log_param("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        mlflow.log_param("cuda_version", torch.version.cuda or "N/A")

        # VRAM check at initialization
        vram_profiler = VRAMProfiler()
        vram_profiler.reset()

        # Test forward pass to check VRAM
        console.print("\n[bold]Testing forward pass (VRAM check)...[/bold]")
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(model_device)

        try:
            test_input = torch.randn(1, 4, 96, 96, 96, device=model_device)
            with torch.cuda.amp.autocast(enabled=config.get("mixed_precision", True)):
                test_output = model(test_input)
            console.print(f"  Input shape: {test_input.shape}")
            console.print(f"  Output shape: {test_output.shape}")
            console.print(f"  VRAM after forward: {vram_profiler.get_peak_mb():.0f} MB")
            mlflow.log_metric("init_vram_peak_mb", vram_profiler.get_peak_mb())
            del test_input, test_output
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                console.print("[bold red]OOM during forward pass test! Model too large for GPU.[/bold red]")
                mlflow.log_param("training_failed", "OOM_at_init")
                mlflow.end_run(status="FAILED")
                sys.exit(1)
            raise

        # Train
        console.print("\n[bold]Starting training...[/bold]")
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name="unet3d",
            output_dir=args.output_dir,
            log_dir=args.log_dir,
        )
        results = trainer.train()

        # Log final results
        if "training_failed" in results:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.log_metric("final_best_val_dice", results["best_val_dice"])
            mlflow.log_param("final_epoch", results["final_epoch"])
            mlflow.log_param("early_stopped", results.get("early_stopped", False))
            console.print(f"\n[bold green]Training complete! Best val dice: {results['best_val_dice']:.4f}[/bold green]")


if __name__ == "__main__":
    main()
