"""Entry point: Train UNETR for brain tumor segmentation.

Usage:
    python scripts/train_unetr.py \
        --config configs/unetr.yaml \
        --base_config configs/base.yaml \
        --output_dir outputs/models/unetr \
        --log_dir outputs/logs/unetr
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
import yaml
from torch.utils.data import DataLoader
from rich.console import Console

from models.unetr import UNETR
from models.param_counter import count_parameters, print_model_summary
from models.adapters import load_pretrained_vit_for_unetr
from data.dataset import BraTSDataset
from data.augmentation import get_train_transforms, get_val_transforms
from training.trainer import Trainer
from training.vram_profiler import VRAMProfiler
from tracking.mlflow_logger import MLflowLogger

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str, base_config_path: str) -> dict:
    """Load and merge model config with base config."""
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)
    config = {**base_config, **model_config}
    return config


def main():
    parser = argparse.ArgumentParser(description="Train UNETR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, default="outputs/logs")
    args = parser.parse_args()

    config = load_config(args.config, args.base_config)

    # Allow TF32
    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    console.print("\n[bold]Building UNETR...[/bold]")
    model = UNETR(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 3),
        input_size=config.get("patch_size", [96, 96, 96])[0],
        patch_size=config.get("patch_size_tokens", 16),
        embedding_dim=config.get("embedding_dim", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        mlp_ratio=config.get("mlp_ratio", 4),
        dropout=config.get("dropout", 0.1),
        drop_path_rate=config.get("drop_path_rate", 0.1),
        use_checkpoint=config.get("gradient_checkpointing", True),
    )

    # Optional pre-trained weight loading
    if config.get("use_pretrained", False):
        console.print("[yellow]Loading pre-trained ViT weights...[/yellow]")
        model = load_pretrained_vit_for_unetr(
            model,
            pretrained_source=config.get("pretrained_source", "vit_base_patch16"),
            in_channels=config.get("in_channels", 4),
        )

    print_model_summary(model, "UNETR")
    total_params = count_parameters(model)
    console.print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Verify no BatchNorm
    for m in model.modules():
        assert not isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)), \
            "BatchNorm detected in UNETR! This is not allowed."

    # Build datasets
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

    with mlflow.start_run(run_name="BrainTumor_UNETR_Training"):
        # Log parameters
        mlflow.log_param("model_name", "unetr")
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("effective_batch_size", config.get("effective_batch_size", 2))
        mlflow.log_param("physical_batch_size", config.get("physical_batch_size", 1))
        mlflow.log_param("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 2))
        mlflow.log_param("patch_size_train", str(config.get("patch_size", [96, 96, 96])))
        mlflow.log_param("patch_size_eval", str(config.get("eval_patch_size", [128, 128, 128])))
        mlflow.log_param("sliding_window_overlap", config.get("sliding_window_overlap", 0.25))
        mlflow.log_param("optimizer", config.get("optimizer", "adamw"))
        mlflow.log_param("learning_rate", config.get("learning_rate", 1e-4))
        mlflow.log_param("weight_decay", config.get("weight_decay", 1e-5))
        mlflow.log_param("warmup_epochs", config.get("warmup_epochs", 15))
        mlflow.log_param("loss_function", "DiceCE")
        mlflow.log_param("normalization", "LayerNorm")
        mlflow.log_param("mixed_precision", config.get("mixed_precision", True))
        mlflow.log_param("gradient_checkpointing", config.get("gradient_checkpointing", True))
        mlflow.log_param("training_cases", len(train_dataset))
        mlflow.log_param("validation_cases", len(val_dataset))
        mlflow.log_param("seed", config.get("seed", 42))
        mlflow.log_param("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        mlflow.log_param("cuda_version", torch.version.cuda or "N/A")
        mlflow.log_param("use_pretrained", config.get("use_pretrained", False))
        mlflow.log_param("embedding_dim", config.get("embedding_dim", 256))
        mlflow.log_param("num_transformer_layers", config.get("num_layers", 6))
        mlflow.log_param("num_attention_heads", config.get("num_heads", 4))

        # VRAM check
        vram_profiler = VRAMProfiler()
        vram_profiler.reset()
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
                console.print("[bold red]OOM during forward pass test![/bold red]")
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
            model_name="unetr",
            output_dir=args.output_dir,
            log_dir=args.log_dir,
        )
        results = trainer.train()

        if "training_failed" in results:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.log_metric("final_best_val_dice", results["best_val_dice"])
            mlflow.log_param("final_epoch", results["final_epoch"])
            mlflow.log_param("early_stopped", results.get("early_stopped", False))
            console.print(f"\n[bold green]Training complete! Best val dice: {results['best_val_dice']:.4f}[/bold green]")


if __name__ == "__main__":
    main()
