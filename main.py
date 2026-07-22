"""Unified CLI entry point for the brain tumor segmentation pipeline.

Supports HPC cluster execution with automatic checkpoint resumption
and a separate evaluation-only mode.

Usage:
    # First run — starts fresh
    python main.py --mode train

    # HPC job killed and requeued — automatically resumes from last checkpoint
    python main.py --mode train

    # Evaluation only — no training, uses existing checkpoints
    python main.py --mode test

    # Train without benchmarks
    python main.py --mode train --skip-benchmarks

    # Custom paths
    python main.py --mode train --config-dir configs/ --output-dir outputs/
    python main.py --mode test --output-dir outputs/
"""
print("Running:", __file__)
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import random


def set_global_seed(seed: int = 42):
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
from models.unetr import UNETR
from models.param_counter import count_parameters, print_model_summary, verify_parameter_parity
from models.adapters import load_pretrained_vit_for_unetr
from data.dataset import BraTSDataset, BenchmarkDataset
from data.augmentation import get_train_transforms, get_val_transforms
from data.deduplication import deduplicate_datasets
from data.splits import generate_splits
from training.trainer import Trainer
from training.vram_profiler import VRAMProfiler
from tracking.mlflow_logger import MLflowLogger
from evaluation.evaluator import BenchmarkEvaluator
from evaluation.report import generate_comparison_report

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint discovery and validation
# ---------------------------------------------------------------------------

def _validate_checkpoint(checkpoint_path: Path) -> bool:
    """Check whether a checkpoint file loads correctly and has required keys.

    Args:
        checkpoint_path: Path to a .pth checkpoint file.

    Returns:
        True if the checkpoint is valid, False otherwise.
    """
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        required_keys = ["epoch", "model_state_dict", "optimizer_state_dict"]
        if not all(k in checkpoint for k in required_keys):
            raise ValueError(
                f"Checkpoint missing required keys: "
                f"{[k for k in required_keys if k not in checkpoint]}"
            )
        return True
    except Exception as e:
        console.print(
            f"[yellow]WARNING: Checkpoint {checkpoint_path} appears corrupted: {e}[/yellow]"
        )
        console.print(
            "[yellow]Falling back to previous checkpoint or starting fresh[/yellow]"
        )
        return False


def find_resume_checkpoint(output_dir: str, model_name: str) -> Optional[str]:
    """Check for an existing checkpoint to resume from.

    Priority order:
        1. best_checkpoint.pth if it exists (most reliable — has full state)
        2. Highest-numbered final_checkpoint_epoch{N}.pth if best is missing
        3. None if no checkpoint exists — start fresh from epoch 0

    Args:
        output_dir: Model-specific output directory (e.g. outputs/models/unet3d).
        model_name: Name for logging purposes.

    Returns:
        Path to checkpoint to resume from, or None to start fresh.
    """
    model_dir = Path(output_dir)
    if not model_dir.exists():
        return None

    # Priority 1: best_checkpoint.pth
    best_ckpt = model_dir / "best_checkpoint.pth"
    if best_ckpt.exists():
        if _validate_checkpoint(best_ckpt):
            console.print(
                f"[green]Found valid best checkpoint for {model_name}: {best_ckpt}[/green]"
            )
            return str(best_ckpt)
        # Best is corrupted — fall through to final checkpoints

    # Priority 2: highest-numbered final_checkpoint_epoch{N}.pth
    final_ckpts = sorted(
        model_dir.glob("final_checkpoint_epoch*.pth"),
        key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1))
        if re.search(r"epoch(\d+)", p.name) else -1,
        reverse=True,
    )
    for ckpt in final_ckpts:
        if _validate_checkpoint(ckpt):
            console.print(
                f"[green]Found valid final checkpoint for {model_name}: {ckpt}[/green]"
            )
            return str(ckpt)

    # Priority 3: no valid checkpoint
    return None


def _is_training_complete(checkpoint_path: str, num_epochs: int) -> dict:
    """Check if training already completed based on checkpoint metadata.

    Args:
        checkpoint_path: Path to the checkpoint to inspect.
        num_epochs: Maximum number of training epochs from config.

    Returns:
        Dict with 'complete' (bool), 'reason' (str), and 'best_val_dice' (float).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", 0)
    best_dice = checkpoint.get("val_dice_mean", 0.0)
    config = checkpoint.get("config", {})

    # Check if early stopping was previously triggered
    es_state = checkpoint.get("early_stopping_state", {})
    if es_state.get("triggered", False):
        return {
            "complete": True,
            "reason": f"early stopping triggered at epoch {es_state.get('triggered_at_epoch', '?')}",
            "best_val_dice": best_dice,
        }

    # If the epoch matches num_epochs - 1, training reached the end
    ckpt_num_epochs = config.get("num_epochs", num_epochs)
    if epoch >= ckpt_num_epochs - 1:
        return {
            "complete": True,
            "reason": f"reached max epochs ({ckpt_num_epochs})",
            "best_val_dice": best_dice,
        }

    return {"complete": False, "reason": "", "best_val_dice": best_dice}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_unet3d(config: dict) -> UNet3D:
    """Build a 3D U-Net model from config.

    Args:
        config: Merged configuration dictionary.

    Returns:
        UNet3D model instance.
    """
    model = UNet3D(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 3),
        base_channels=config.get("base_channels", 60),
        groups=config.get("groups", 4),
        dropout_rates=config.get("dropout_rates", [0.0, 0.0, 0.0, 0.0]),
        drop_path_rate=config.get("drop_path_rate", 0.0),
    )
    return model


def build_unetr(config: dict) -> UNETR:
    """Build a UNETR model from config.

    Args:
        config: Merged configuration dictionary.

    Returns:
        UNETR model instance.
    """
    model = UNETR(
        in_channels=config.get("in_channels", 4),
        out_channels=config.get("out_channels", 3),
        input_size=config.get("patch_size", [128, 128, 128])[0],
        patch_size=config.get("patch_size_tokens", 16),
        embedding_dim=config.get("embedding_dim", 384),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 8),
        mlp_ratio=config.get("mlp_ratio", 4),
        dropout=config.get("dropout", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.0),
        use_checkpoint=config.get("gradient_checkpointing", False),
    )

    # Optional pre-trained weight loading
    if config.get("use_pretrained", False):
        console.print("[yellow]Loading pre-trained ViT weights...[/yellow]")
        model = load_pretrained_vit_for_unetr(
            model,
            pretrained_source=config.get("pretrained_source", "vit_base_patch16"),
            in_channels=config.get("in_channels", 4),
        )

    return model





# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict, manifest_dir: str) -> tuple:
    """Build training and validation DataLoaders from manifests.

    Args:
        config: Merged configuration dictionary.
        manifest_dir: Directory containing train_manifest.csv and val_manifest.csv.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
    """
    train_manifest = Path(manifest_dir) / "train_manifest.csv"
    val_manifest = Path(manifest_dir) / "val_manifest.csv"

    if not train_manifest.exists():
        console.print(f"[red]ERROR: Train manifest not found: {train_manifest}[/red]")
        console.print("[yellow]Run deduplication first or use --mode train[/yellow]")
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

    num_workers = config.get("num_workers", 8)
    use_persistent = config.get("persistent_workers", True) and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("physical_batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", True),
        drop_last=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", True),
        persistent_workers=use_persistent,
    )

    return train_loader, val_loader, train_dataset, val_dataset


# ---------------------------------------------------------------------------
# VRAM forward-pass check
# ---------------------------------------------------------------------------

def vram_forward_check(model, config: dict, model_name: str):
    """Run a test forward pass to verify VRAM fits.

    Args:
        model: The model to test.
        config: Configuration dict (for mixed_precision flag).
        model_name: Name for error messages.

    Raises:
        SystemExit: If OOM occurs during the test.
    """
    vram_profiler = VRAMProfiler()
    vram_profiler.reset()

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(model_device)

    try:
        patch_size = tuple(config.get("patch_size", [128, 128, 128]))
        test_input = torch.randn(1, 4, *patch_size, device=model_device)
        with torch.amp.autocast("cuda", enabled=config.get("mixed_precision", True)):
            test_output = model(test_input)
        console.print(f"  Input shape: {test_input.shape}")
        console.print(f"  Output shape: {test_output.shape}")
        console.print(f"  VRAM after forward: {vram_profiler.get_peak_mb():.0f} MB")
        mlflow.log_metric("init_vram_peak_mb", vram_profiler.get_peak_mb())
        del test_input, test_output
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(
                f"[bold red]OOM during {model_name} forward pass test! "
                f"Model too large for GPU.[/bold red]"
            )
            mlflow.log_param("training_failed", "OOM_at_init")
            mlflow.end_run(status="FAILED")
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Training pipeline for a single model
# ---------------------------------------------------------------------------

def train_single_model(
    model_name: str,
    model_builder,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_dataset,
    val_dataset,
    output_dir: str,
    log_dir: str,
    mlflow_run_name: str,
    extra_mlflow_params: Optional[dict] = None,
):
    """Train a single model with automatic checkpoint resumption.

    Args:
        model_name: Short name ('unet3d' or 'unetr').
        model_builder: Callable that returns the model.
        config: Merged config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        train_dataset: Training dataset (for length logging).
        val_dataset: Validation dataset (for length logging).
        output_dir: Model checkpoint output directory.
        log_dir: Training log directory.
        mlflow_run_name: MLflow run name.
        extra_mlflow_params: Additional model-specific MLflow params.
    """
    model_output_dir = str(Path(output_dir) / f"models/{model_name}")
    model_log_dir = str(Path(output_dir) / f"logs/{model_name}")

    # --- Auto-resume checkpoint discovery ---
    resume_ckpt = find_resume_checkpoint(model_output_dir, model_name)
    start_epoch = 0

    if resume_ckpt is not None:
        # Check if training is already complete
        completion = _is_training_complete(
            resume_ckpt, config.get("num_epochs", 300)
        )
        if completion["complete"]:
            console.print(
                f"[bold green]{model_name} training already complete "
                f"({completion['reason']}, best_val_dice={completion['best_val_dice']:.4f}), "
                f"skipping[/bold green]"
            )
            return

    # --- Build model ---
    console.print(f"\n[bold]Building {model_name.upper()}...[/bold]")
    model = model_builder(config)
    print_model_summary(model, model_name.upper())
    total_params = count_parameters(model)
    console.print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Verify no BatchNorm
    for m in model.modules():
        assert not isinstance(
            m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ), f"BatchNorm detected in {model_name}! This is not allowed."

    # --- MLflow run ---
    with mlflow.start_run(run_name=mlflow_run_name):
        # Common parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("effective_batch_size", config.get("effective_batch_size", 2))
        mlflow.log_param("physical_batch_size", config.get("physical_batch_size", 1))
        mlflow.log_param(
            "gradient_accumulation_steps",
            config.get("gradient_accumulation_steps", 2),
        )
        mlflow.log_param(
            "patch_size_train", str(config.get("patch_size", [96, 96, 96]))
        )
        mlflow.log_param(
            "patch_size_eval", str(config.get("eval_patch_size", [128, 128, 128]))
        )
        mlflow.log_param(
            "sliding_window_overlap", config.get("sliding_window_overlap", 0.25)
        )
        mlflow.log_param("optimizer", config.get("optimizer", "adamw"))
        mlflow.log_param("learning_rate", config.get("learning_rate", 3e-4))
        mlflow.log_param("weight_decay", config.get("weight_decay", 1e-5))
        mlflow.log_param("warmup_epochs", config.get("warmup_epochs", 10))
        mlflow.log_param("loss_function", "DiceCE")
        mlflow.log_param("mixed_precision", config.get("mixed_precision", True))
        mlflow.log_param("training_cases", len(train_dataset))
        mlflow.log_param("validation_cases", len(val_dataset))
        mlflow.log_param("seed", config.get("seed", 42))
        mlflow.log_param(
            "gpu",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        )
        mlflow.log_param("cuda_version", torch.version.cuda or "N/A")

        # Model-specific parameters
        if extra_mlflow_params:
            for key, value in extra_mlflow_params.items():
                mlflow.log_param(key, value)

        # Resume tagging
        if resume_ckpt is not None:
            # Load checkpoint to determine start_epoch before logging
            # (we need to build the trainer first)
            mlflow.set_tag("resumed", "true")
            mlflow.set_tag("original_run_name", mlflow_run_name)
        else:
            mlflow.set_tag("resumed", "false")

        # VRAM check
        console.print(f"\n[bold]Testing forward pass (VRAM check)...[/bold]")
        vram_forward_check(model, config, model_name)

        # Build trainer
        console.print(f"\n[bold]Initializing trainer for {model_name}...[/bold]")
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            output_dir=model_output_dir,
            log_dir=model_log_dir,
        )

        # Resume from checkpoint if available
        if resume_ckpt is not None:
            start_epoch = trainer.load_checkpoint(resume_ckpt)
            console.print(
                f"[bold cyan]Resuming {model_name} from epoch {start_epoch}[/bold cyan]"
            )
            mlflow.log_param("resumed_from_epoch", start_epoch)
            mlflow.set_tag("resumed_from_epoch", str(start_epoch))
        else:
            console.print(
                f"[bold]Starting {model_name} training from scratch[/bold]"
            )
            mlflow.log_param("resumed_from_epoch", 0)

        # Train
        results = trainer.train(start_epoch=start_epoch)

        # Log final results
        if "training_failed" in results:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.log_metric("final_best_val_dice", results["best_val_dice"])
            mlflow.log_param("final_epoch", results["final_epoch"])
            mlflow.log_param("early_stopped", results.get("early_stopped", False))
            console.print(
                f"\n[bold green]{model_name} training complete! "
                f"Best val dice: {results['best_val_dice']:.4f}[/bold green]"
            )


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

def run_benchmarks(output_dir: str, base_config: dict):
    """Run benchmark evaluation on UPenn-GBM and BraTS-SSA datasets.

    Loads best checkpoints for available models and evaluates using
    BenchmarkEvaluator with sliding window inference.

    Args:
        output_dir: Root output directory.
        base_config: Base configuration dictionary.
    """
    model_ckpts = {
        "unet3d": Path(output_dir) / "models/unet3d/best_checkpoint.pth",
        "unetr": Path(output_dir) / "models/unetr/best_checkpoint.pth",
    }

    # Determine which models have checkpoints
    available_models = {}
    for model_name, ckpt_path in model_ckpts.items():
        if ckpt_path.exists():
            available_models[model_name] = ckpt_path
        else:
            console.print(
                f"[yellow]{model_name} checkpoint not found at {ckpt_path} — skipping[/yellow]"
            )

    if not available_models:
        console.print("[yellow]No model checkpoints available — skipping all benchmarks[/yellow]")
        return

    eval_patch_size = tuple(base_config.get("eval_patch_size", [128, 128, 128]))
    overlap = base_config.get("sliding_window_overlap", 0.25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which benchmarks are available
    data_config = base_config.get("data", {})
    upenn_gbm_dir = Path(data_config.get("upenn_gbm_dir", "data/raw/UPenn-GBM"))
    ssa_dir = Path(data_config.get("ssa_dir", "data/raw/brats_ssa"))

    benchmarks = []
    if upenn_gbm_dir.exists() and any(upenn_gbm_dir.iterdir()):
        benchmarks.append(("upenn_gbm", upenn_gbm_dir, "UPennGBM"))
    else:
        console.print(
            f"[yellow]UPenn-GBM benchmark data not found at {upenn_gbm_dir} — skipping UPenn-GBM benchmark[/yellow]"
        )

    if ssa_dir.exists() and any(ssa_dir.iterdir()):
        benchmarks.append(("ssa", ssa_dir, "SSA"))
    else:
        console.print(
            f"[yellow]BraTS-SSA benchmark data not found at {ssa_dir} — skipping SSA benchmark[/yellow]"
        )

    if not benchmarks:
        console.print("[yellow]No benchmark datasets available — skipping all benchmarks[/yellow]")
        return

    for bench_type, bench_dir, bench_name in benchmarks:
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]Benchmarking on {bench_name}...[/bold]")
        console.print(f"[bold]{'='*60}[/bold]")

        from scripts.benchmark import load_model_from_checkpoint

        # Load all available models
        console.print(f"\n[bold]Loading models for {bench_name}...[/bold]")
        models = {}
        val_dices = {}
        for model_name, ckpt_path in available_models.items():
            model, val_dice = load_model_from_checkpoint(str(ckpt_path), model_name)
            models[model_name] = model
            val_dices[model_name] = val_dice

        # Load benchmark dataset
        console.print(f"\n[bold]Loading {bench_name} benchmark dataset...[/bold]")
        dataset = BenchmarkDataset(
            data_dir=str(bench_dir),
            benchmark_type=bench_type,
        )
        console.print(f"  Found {len(dataset)} cases")

        if len(dataset) == 0:
            console.print(
                f"[yellow]{bench_name} dataset has 0 valid cases — skipping[/yellow]"
            )
            for m in models.values():
                del m
            torch.cuda.empty_cache()
            continue

        # Evaluate each model
        all_results = {}
        model_display_names = {"unet3d": "UNet3D", "unetr": "UNETR"}
        for model_name in available_models:
            display_name = model_display_names[model_name]
            console.print(f"\n[bold]Evaluating {display_name} on {bench_name}...[/bold]")
            evaluator = BenchmarkEvaluator(
                model=models[model_name],
                model_name=model_name,
                device=device,
                eval_patch_size=eval_patch_size,
                overlap=overlap,
                sw_batch_size=1,
                use_amp=True,
            )
            results = evaluator.evaluate_dataset(
                dataset=dataset,
                output_dir=str(Path(output_dir) / f"reports/{bench_type}"),
                mlflow_run_name=f"BrainTumor_{display_name}_Benchmark_{bench_name}",
                val_dice_mean=val_dices[model_name],
            )
            all_results[model_name] = results

            # Free VRAM between models
            del models[model_name], evaluator
            torch.cuda.empty_cache()

        # Print comparison
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]{bench_name} Benchmark Results[/bold]")
        console.print(f"[bold]{'='*60}[/bold]")
        for model_name, display_name in model_display_names.items():
            if model_name in all_results:
                r = all_results[model_name]
                console.print(
                    f"  {display_name:12s} Dice Mean: {r.get('dice_mean', 0.0):.4f}  "
                    f"HD95 Mean: {r.get('hd95_mean', 0.0):.2f}"
                )

        if bench_type == "upenn_gbm":
            console.print(f"\n  Per tumor type:")
            for model_name, display_name in model_display_names.items():
                if model_name in all_results:
                    r = all_results[model_name]
                    console.print(
                        f"    {display_name:12s} GBM: {r.get('dice_mean_GBM', 0.0):.4f}  "
                        f"LGG: {r.get('dice_mean_LGG', 0.0):.4f}"
                    )


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def mode_train(args):
    """Execute the full training pipeline with auto-resume.

    Steps:
        1. Deduplication and split generation
        2. Train 3D U-Net (with auto-resume)
        3. Train UNETR (with auto-resume)
        4. Benchmark evaluation (unless --skip-benchmarks)
        5. Final comparison report
    """
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)

    base_config_path = config_dir / "base.yaml"
    unet_config_path = config_dir / "unet3d.yaml"
    unetr_config_path = config_dir / "unetr.yaml"


    # Validate config files exist
    for path, name in [
        (base_config_path, "base.yaml"),
        (unet_config_path, "unet3d.yaml"),
        (unetr_config_path, "unetr.yaml"),
    ]:
        if not path.exists():
            console.print(f"[red]ERROR: Config file not found: {path}[/red]")
            sys.exit(1)

    # Load base config for shared settings
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Allow TF32 for Turing GPUs
    if base_config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri=base_config.get("mlflow_tracking_uri", "outputs/mlruns"),
        experiment_name=base_config.get("experiment_name", "BrainTumorBenchmark"),
    )

    # Create output directories
    data_config = base_config.get("data", {})
    manifests_dir = str(output_dir / "logs")
    (output_dir / "models/unet3d").mkdir(parents=True, exist_ok=True)
    (output_dir / "models/unetr").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs/unet3d").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs/unetr").mkdir(parents=True, exist_ok=True)
    (output_dir / "mlruns").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports/upenn_gbm").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports/ssa").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Deduplication and split generation
    # -----------------------------------------------------------------------
    console.print("\n[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold][1/6] Deduplication and manifest generation[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")

    brats2021_dir = data_config.get("brats2021_dir", "data/raw/brats2021")
    brats2024_dir = data_config.get("brats2024_dir", "data/raw/brats2024")

    if not Path(brats2021_dir).exists():
        console.print(f"[red]ERROR: BraTS 2021 directory not found: {brats2021_dir}[/red]")
        sys.exit(1)
    if not Path(brats2024_dir).exists():
        console.print(f"[red]ERROR: BraTS 2024 directory not found: {brats2024_dir}[/red]")
        sys.exit(1)

    with mlflow.start_run(run_name="DataPreparation"):
        mlflow.log_param("seed", base_config.get("seed", 42))
        mlflow.log_param("brats2021_dir", brats2021_dir)
        mlflow.log_param("brats2024_dir", brats2024_dir)

        dedup_result = deduplicate_datasets(
            brats2021_dir=brats2021_dir,
            brats2024_dir=brats2024_dir,
            output_dir=manifests_dir,
        )

        mlflow.log_artifact(dedup_result["report_path"])
        mlflow.log_param("duplicates_found", dedup_result["report"]["duplicates_found"])
        mlflow.log_param("final_pool_size", dedup_result["report"]["final_pool_size"])

        seed = base_config.get("seed", 42)
        train_df, val_df = generate_splits(
            valid_cases_2021=dedup_result["valid_cases_2021"],
            valid_cases_2024=dedup_result["valid_cases_2024"],
            output_dir=manifests_dir,
            seed=seed,
        )

        train_manifest_path = Path(manifests_dir) / "train_manifest.csv"
        val_manifest_path = Path(manifests_dir) / "val_manifest.csv"
        mlflow.log_artifact(str(train_manifest_path))
        mlflow.log_artifact(str(val_manifest_path))
        mlflow.log_param("training_cases", len(train_df))
        mlflow.log_param("validation_cases", len(val_df))
        mlflow.log_param(
            "train_cases_brats2021",
            len(train_df[train_df["dataset_origin"] == "brats2021"]),
        )
        mlflow.log_param(
            "train_cases_brats2024",
            len(train_df[train_df["dataset_origin"] == "brats2024"]),
        )

    console.print("[bold green]✓ Deduplication and split generation complete![/bold green]")

    # -----------------------------------------------------------------------
    # Step 2: Train 3D U-Net
    # -----------------------------------------------------------------------
    if args.model is None or args.model == "unet3d":
        console.print("\n[bold]" + "=" * 60 + "[/bold]")
        console.print("[bold][2/6] Training 3D U-Net[/bold]")
        console.print("[bold]" + "=" * 60 + "[/bold]")

        unet_config = load_config(str(unet_config_path), str(base_config_path))
        train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
            unet_config, manifests_dir
        )
        console.print(f"  Train: {len(train_dataset)} cases")
        console.print(f"  Val: {len(val_dataset)} cases")

        train_single_model(
            model_name="unet3d",
            model_builder=build_unet3d,
            config=unet_config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=str(output_dir),
            log_dir=str(output_dir / "logs/unet3d"),
            mlflow_run_name="BrainTumor_UNet3D_Training",
            extra_mlflow_params={
                "normalization": "GroupNorm",
                "gradient_checkpointing": False,
            },
        )

    # -----------------------------------------------------------------------
    # Step 3: Train UNETR
    # -----------------------------------------------------------------------
    if args.model is None or args.model == "unetr":
        console.print("\n[bold]" + "=" * 60 + "[/bold]")
        console.print("[bold][3/6] Training UNETR[/bold]")
        console.print("[bold]" + "=" * 60 + "[/bold]")

        unetr_config = load_config(str(unetr_config_path), str(base_config_path))
        train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
            unetr_config, manifests_dir
        )
        console.print(f"  Train: {len(train_dataset)} cases")
        console.print(f"  Val: {len(val_dataset)} cases")

        train_single_model(
            model_name="unetr",
            model_builder=build_unetr,
            config=unetr_config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=str(output_dir),
            log_dir=str(output_dir / "logs/unetr"),
            mlflow_run_name="BrainTumor_UNETR_Training",
            extra_mlflow_params={
                "normalization": "LayerNorm",
                "gradient_checkpointing": unetr_config.get("gradient_checkpointing", True),
                "use_pretrained": unetr_config.get("use_pretrained", False),
                "embedding_dim": unetr_config.get("embedding_dim", 256),
                "num_transformer_layers": unetr_config.get("num_layers", 6),
                "num_attention_heads": unetr_config.get("num_heads", 4),
            },
        )


    # -----------------------------------------------------------------------
    # Step 5: Benchmark evaluation
    # -----------------------------------------------------------------------
    if args.skip_benchmarks:
        console.print(
            "\n[yellow]--skip-benchmarks set — skipping benchmark evaluation[/yellow]"
        )
    else:
        console.print("\n[bold]" + "=" * 60 + "[/bold]")
        console.print("[bold][5/6] Benchmark evaluation[/bold]")
        console.print("[bold]" + "=" * 60 + "[/bold]")

        unet_ckpt = output_dir / "models/unet3d/best_checkpoint.pth"
        unetr_ckpt = output_dir / "models/unetr/best_checkpoint.pth"

        missing = []
        if not unet_ckpt.exists():
            missing.append(f"UNet3D: {unet_ckpt}")
        if not unetr_ckpt.exists():
            missing.append(f"UNETR: {unetr_ckpt}")

        if missing:
            console.print(
                f"[yellow]WARNING: Missing checkpoints for benchmarking: "
                f"{', '.join(missing)}. Skipping benchmarks.[/yellow]"
            )
        else:
            run_benchmarks(str(output_dir), base_config)

    # -----------------------------------------------------------------------
    # Step 6: Final comparison report
    # -----------------------------------------------------------------------
    console.print("\n[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold][6/6] Final comparison report[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")

    generate_comparison_report(
        reports_dir=str(output_dir / "reports"),
        output_path=str(output_dir / "reports/final_comparison.csv"),
    )

    console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
    console.print("[bold green] ALL DONE.[/bold green]")
    console.print(
        f"[bold green] View results: mlflow ui "
        f"--backend-store-uri {output_dir / 'mlruns'}[/bold green]"
    )
    console.print("[bold green]" + "=" * 60 + "[/bold green]")


def mode_test(args):
    """Execute evaluation-only mode — no training.

    Requires that best_checkpoint.pth exists for both models.
    Runs available benchmarks and generates comparison report.
    """
    console.print("\n[bold]Running evaluation-only mode — no training will occur[/bold]")

    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    base_config_path = config_dir / "base.yaml"

    if not base_config_path.exists():
        console.print(f"[red]ERROR: Base config not found: {base_config_path}[/red]")
        sys.exit(1)

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Initialize MLflow
    mlflow_logger = MLflowLogger(
        tracking_uri=base_config.get("mlflow_tracking_uri", "outputs/mlruns"),
        experiment_name=base_config.get("experiment_name", "BrainTumorBenchmark"),
    )

    # Validate checkpoints — warn about missing, but proceed if at least one exists
    model_ckpts = {
        "UNet3D": output_dir / "models/unet3d/best_checkpoint.pth",
        "UNETR": output_dir / "models/unetr/best_checkpoint.pth",
    }

    found = []
    missing = []
    for name, ckpt in model_ckpts.items():
        if ckpt.exists():
            found.append(name)
            console.print(f"  [green]✓ {name}: {ckpt}[/green]")
        else:
            missing.append(name)
            console.print(f"  [yellow]✗ {name}: not found at {ckpt} — will be skipped[/yellow]")

    if not found:
        console.print(
            "[red]ERROR: No model checkpoints found. "
            "Run with --mode train first, or check --output-dir path[/red]"
        )
        sys.exit(1)

    console.print(f"\n[green]✓ {len(found)} model(s) available for evaluation[/green]")

    # Run benchmarks
    run_benchmarks(str(output_dir), base_config)

    # Generate comparison report
    console.print("\n[bold]Generating final comparison report...[/bold]")
    generate_comparison_report(
        reports_dir=str(output_dir / "reports"),
        output_path=str(output_dir / "reports/final_comparison.csv"),
    )

    console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
    console.print("[bold green] EVALUATION COMPLETE.[/bold green]")
    console.print(
        f"[bold green] View results: mlflow ui "
        f"--backend-store-uri {output_dir / 'mlruns'}[/bold green]"
    )
    console.print("[bold green]" + "=" * 60 + "[/bold green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation: 3D U-Net vs UNETR — "
        "Unified pipeline with HPC auto-resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --mode train                      # Full pipeline (fresh or auto-resume)
  python main.py --mode train --skip-benchmarks    # Train only, skip benchmarks
  python main.py --mode test                       # Evaluation only (requires checkpoints)
  python main.py --mode test --output-dir outputs/ # Custom output directory
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="'train' runs deduplication+split+training for all models with auto-resume. "
        "'test' runs benchmark evaluation only, using existing checkpoints, no training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet3d", "unetr"],
        default=None,
        help="Train only the selected model. If omitted, all models are trained.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/",
        help="Directory containing base.yaml, unet3d.yaml, unetr.yaml (default: configs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Root output directory for checkpoints, logs, reports (default: outputs/)",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        default=False,
        help="Skip benchmark evaluation even if data is available (train mode only)",
    )

    args = parser.parse_args()

    console.print("\n[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold] Brain Tumor Segmentation — 3D U-Net vs UNETR[/bold]")
    console.print(f"[bold] Mode: {args.mode.upper()}[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "test":
        mode_test(args)


if __name__ == "__main__":
    main()
