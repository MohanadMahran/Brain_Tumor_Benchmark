"""Unified Trainer class for brain tumor segmentation.

Works identically for both 3D U-Net and UNETR.
Handles:
    - Gradient accumulation
    - Mixed precision (FP16)
    - VRAM monitoring and OOM handling
    - MLflow logging
    - Early stopping
    - Checkpoint management
    - Mixup augmentation
    - Test-time augmentation (TTA) for validation
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console

from training.losses import DiceCELoss
from training.metrics import compute_all_metrics
from training.early_stopping import EarlyStopping
from training.vram_profiler import VRAMProfiler

console = Console()
logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(model: nn.Module, config: dict) -> tuple:
    """Build optimizer and learning rate scheduler.

    Schedule: Linear warmup -> Cosine annealing

    Args:
        model: PyTorch model.
        config: Configuration dict with optimizer parameters.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    num_epochs = config.get("num_epochs", 300)
    warmup_epochs = config.get("warmup_epochs", 10)
    min_lr = config.get("min_lr", 1e-6)
    base_lr = config["learning_rate"]

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Linear warmup from lr/100 to full lr
            return (epoch / warmup_epochs) * (1.0 - 1.0 / 100) + 1.0 / 100
        elif warmup_epochs > 0:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr / base_lr, cosine_decay)
        else:
            # No warmup — cosine annealing from epoch 0
            progress = epoch / max(num_epochs, 1)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr / base_lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


class Trainer:
    """Unified trainer for brain tumor segmentation models.

    Handles the complete training loop with:
    - Gradient accumulation (physical_batch=1, effective_batch=2)
    - Mixed precision (FP16 on Turing GPUs)
    - Early stopping on validation Dice
    - MLflow experiment tracking
    - VRAM monitoring with OOM handling
    - Mixup augmentation
    - TTA at validation time

    Args:
        model: Neural network model (UNet3D or UNETR).
        config: Training configuration dictionary.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        model_name: Name for logging and checkpointing.
        output_dir: Directory for saving checkpoints.
        log_dir: Directory for training logs.
    """

    # Wall-time safety: save and exit 10 minutes before Slurm kills the job.
    # Default 24h (86400s) minus 600s = 85800s (~23h50m).
    DEFAULT_WALL_TIME_LIMIT_S = 86400
    WALL_TIME_SAFETY_MARGIN_S = 600  # 10 minutes

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        output_dir: str,
        log_dir: str,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = DiceCELoss(
            smooth=1.0,
            dice_weight=0.5,
            ce_weight=0.5,
            label_smoothing=config.get("label_smoothing", 0.0),
        )

        # Optimizer and scheduler
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(model, config)

        # Mixed precision — use non-deprecated API
        self.use_amp = config.get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 30),
            mode="max",
        )

        # VRAM profiler
        self.vram_profiler = VRAMProfiler()

        # Training state
        self.best_val_dice = 0.0
        self.best_val_metrics = {}
        self.current_epoch = 0

        # Wall-time safety: record when training started
        self._training_start_time = time.monotonic()
        wall_limit = config.get("wall_time_limit_s", self.DEFAULT_WALL_TIME_LIMIT_S)
        self._wall_time_deadline = self._training_start_time + wall_limit - self.WALL_TIME_SAFETY_MARGIN_S

        # Config shortcuts
        self.num_epochs = config.get("num_epochs", 300)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 2)
        self.log_every_n = config.get("log_every_n_epochs", 10)
        self.mixup_prob = config.get("mixup_prob", 0.0)
        self.mixup_alpha = config.get("mixup_alpha", 0.2)
        self.tta_enabled = config.get("tta_enabled", False)

        # Parameter count for logging
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Trainer initialized for {model_name} on {self.device}. "
            f"Epochs={self.num_epochs}, GradAccum={self.grad_accum_steps}, "
            f"AMP={self.use_amp}, Params={self.total_params:,} ({self.total_params/1e6:.2f}M)"
        )

    def train(self, start_epoch: int = 0) -> Dict:
        """Run full training loop.

        Args:
            start_epoch: Epoch to resume from (0 for fresh start).

        Returns:
            Dict with final training results and best metrics.
        """
        self.vram_profiler.reset()

        # Print training summary header
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]{self.model_name.upper()} Training Summary[/bold]")
        console.print(f"[bold]{'='*60}[/bold]")
        console.print(f"  Parameters:       {self.total_params:,} ({self.total_params/1e6:.2f}M)")
        console.print(f"  Training samples: {len(self.train_loader.dataset)}")
        console.print(f"  Validation samples: {len(self.val_loader.dataset)}")
        console.print(f"  Epochs:           {self.num_epochs}")
        console.print(f"  Batch size:       {self.config.get('physical_batch_size', 1)}")
        console.print(f"  Grad accum:       {self.grad_accum_steps}")
        console.print(f"  Learning rate:    {self.config.get('learning_rate', 1e-3)}")
        console.print(f"  Weight decay:     {self.config.get('weight_decay', 0)}")
        console.print(f"  Mixed precision:  {self.use_amp}")
        console.print(f"  Mixup prob:       {self.mixup_prob}")
        console.print(f"  TTA:              {self.tta_enabled}")
        console.print(f"[bold]{'='*60}[/bold]\n")

        try:
            wall_time_exit = False
            for epoch in range(start_epoch, self.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Wall-time safety check BEFORE starting a new epoch
                if self._should_stop_for_wall_time():
                    console.print(
                        f"[bold yellow]Wall-time safety: approaching deadline, "
                        f"saving checkpoint and exiting cleanly after epoch {epoch - 1}[/bold yellow]"
                    )
                    wall_time_exit = True
                    break

                # Training epoch
                train_loss, train_dice = self._train_epoch(epoch)

                # Validation (every log_every_n epochs or at end)
                val_metrics = None
                val_loss = None
                if (epoch + 1) % self.log_every_n == 0 or epoch == self.num_epochs - 1:
                    val_metrics = self._validate_epoch(epoch)
                    val_dice_mean = val_metrics.get("dice_mean", 0.0)
                    val_loss = val_metrics.get("val_loss", 0.0)

                    # Log to MLflow
                    self._log_metrics(epoch, train_loss, val_metrics, train_dice)

                    # Print epoch summary
                    epoch_time = time.time() - epoch_start
                    lr = self.scheduler.get_last_lr()[0]
                    console.print(
                        f"[bold]Epoch {epoch+1:03d}/{self.num_epochs}[/bold] | "
                        f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice_mean:.4f} | "
                        f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
                    )

                    # Check for best model
                    if val_dice_mean > self.best_val_dice:
                        self.best_val_dice = val_dice_mean
                        self.best_val_metrics = val_metrics
                        self._save_checkpoint(epoch, is_best=True)
                        console.print(
                            f"  [green]★ New best! Val Dice: {val_dice_mean:.4f}[/green]"
                        )

                    # Early stopping check
                    if self.early_stopping(val_dice_mean, epoch):
                        logger.info(f"Early stopping at epoch {epoch}")
                        mlflow.log_param("early_stopping_triggered", True)
                        mlflow.log_param("early_stopping_epoch", epoch)
                        break

                # Update scheduler
                self.scheduler.step()

            # Save final checkpoint
            self._save_checkpoint(self.current_epoch, is_best=False)

            if wall_time_exit:
                console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
                console.print(
                    f"[bold yellow]{self.model_name.upper()} preempted by wall-time safety at epoch {self.current_epoch}. "
                    f"Best Val Dice: {self.best_val_dice:.4f}[/bold yellow]"
                )
                console.print(f"[bold yellow]{'='*60}[/bold yellow]")
            else:
                console.print(f"\n[bold green]{'='*60}[/bold green]")
                console.print(
                    f"[bold green]{self.model_name.upper()} training complete! "
                    f"Best Val Dice: {self.best_val_dice:.4f}[/bold green]"
                )
                console.print(f"[bold green]{'='*60}[/bold green]")

            return {
                "best_val_dice": self.best_val_dice,
                "best_val_metrics": self.best_val_metrics,
                "final_epoch": self.current_epoch,
                "early_stopped": self.early_stopping.triggered,
                "wall_time_exit": wall_time_exit,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._handle_oom(e)
                return {"training_failed": "OOM", "oom_epoch": self.current_epoch}
            raise

    def _train_epoch(self, epoch: int) -> tuple:
        """Run a single training epoch with gradient accumulation.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, average_dice).
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            # Mixup augmentation
            if self.mixup_prob > 0 and np.random.random() < self.mixup_prob:
                image, label = self._apply_mixup(image, label)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(image)
                loss = self.criterion(logits, label)
                loss = loss / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()
            accum_count += 1

            # Optimizer step after accumulation
            if accum_count >= self.grad_accum_steps:
                if self.config.get("clip_grad", 0.0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accum_count = 0

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # Compute training Dice for this batch (no grad needed)
            with torch.no_grad():
                pred_probs = torch.sigmoid(logits)
                pred_binary = (pred_probs > 0.5).float()
                # Per-channel Dice
                intersection = (pred_binary * label).sum(dim=(0, 2, 3, 4))
                union = pred_binary.sum(dim=(0, 2, 3, 4)) + label.sum(dim=(0, 2, 3, 4))
                dice = (2.0 * intersection + 1.0) / (union + 1.0)
                total_dice += dice.mean().item()

        # Handle remaining accumulated gradients
        if accum_count > 0:
            if self.config.get("clip_grad", 0.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        avg_dice = total_dice / max(num_batches, 1)
        return avg_loss, avg_dice

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation with optional TTA.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of validation metrics including val_loss.
        """
        self.model.eval()
        all_metrics = []
        metrics_by_origin = {"brats2021": [], "brats2024": []}
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                image = batch["image"].to(self.device, non_blocking=True)
                label = batch["label"].to(self.device, non_blocking=True)
                origin = batch.get("dataset_origin", ["unknown"])[0]

                # Compute val_loss regardless of TTA mode
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(image)
                    val_loss = self.criterion(logits, label)
                    total_val_loss += val_loss.item()
                    num_val_batches += 1

                # Get predictions (with optional TTA)
                if self.tta_enabled:
                    pred = self._tta_predict(image)
                else:
                    pred = torch.sigmoid(logits)

                # Threshold predictions
                pred_binary = (pred > 0.5).cpu().numpy()
                target_binary = label.cpu().numpy()

                # Compute metrics per sample in batch
                for b in range(pred_binary.shape[0]):
                    metrics = compute_all_metrics(pred_binary[b], target_binary[b])
                    all_metrics.append(metrics)
                    if origin in metrics_by_origin:
                        metrics_by_origin[origin].append(metrics["dice_mean"])

        # Aggregate
        result = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                result[key] = float(np.mean(values))

        # Add validation loss
        result["val_loss"] = total_val_loss / max(num_val_batches, 1)

        # Per-origin breakdown
        for origin, scores in metrics_by_origin.items():
            if scores:
                result[f"dice_{origin}"] = float(np.mean(scores))

        return result

    def _tta_predict(self, image: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation: average over 8 flip orientations.

        Args:
            image: Input tensor of shape (B, C, D, H, W).

        Returns:
            Averaged prediction probabilities.
        """
        predictions = []
        # All 8 combinations of flips along 3 axes
        for flip_d in [False, True]:
            for flip_h in [False, True]:
                for flip_w in [False, True]:
                    x = image.clone()
                    if flip_d:
                        x = torch.flip(x, dims=[2])
                    if flip_h:
                        x = torch.flip(x, dims=[3])
                    if flip_w:
                        x = torch.flip(x, dims=[4])

                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        pred = torch.sigmoid(self.model(x))

                    # Reverse flips
                    if flip_w:
                        pred = torch.flip(pred, dims=[4])
                    if flip_h:
                        pred = torch.flip(pred, dims=[3])
                    if flip_d:
                        pred = torch.flip(pred, dims=[2])

                    predictions.append(pred)

        return torch.stack(predictions).mean(dim=0)

    def _apply_mixup(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple:
        """Apply mixup augmentation.

        Blends two samples proportionally (random Beta-distributed lambda).

        Args:
            image: Batch images (B, C, D, H, W).
            label: Batch labels (B, C, D, H, W).

        Returns:
            Tuple of (mixed_image, mixed_label).
        """
        batch_size = image.shape[0]
        if batch_size < 2:
            return image, label

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # Shuffle indices
        indices = torch.randperm(batch_size, device=image.device)
        mixed_image = lam * image + (1 - lam) * image[indices]
        mixed_label = lam * label + (1 - lam) * label[indices]
        return mixed_image, mixed_label

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict,
        train_dice: float = 0.0,
    ):
        """Log metrics to MLflow.

        Args:
            epoch: Current epoch.
            train_loss: Average training loss.
            val_metrics: Validation metrics dict.
            train_dice: Average training Dice.
        """
        try:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)
            # VRAM profiling
            self.vram_profiler.log_to_mlflow(prefix="", step=epoch)
        except Exception as e:
            logger.warning(f"MLflow logging failed at epoch {epoch}: {e}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best checkpoint.
        """
        from models.param_counter import count_parameters

        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_dice_mean": self.best_val_dice,
            "val_dice_ET": self.best_val_metrics.get("dice_ET", 0.0),
            "val_dice_TC": self.best_val_metrics.get("dice_TC", 0.0),
            "val_dice_WT": self.best_val_metrics.get("dice_WT", 0.0),
            "config": self.config,
            "total_parameters": count_parameters(self.model),
            "training_cases": len(self.train_loader.dataset),
            "seed": self.config.get("seed", 42),
            "early_stopping_state": self.early_stopping.state_dict(),
            # Random states for exact reproducibility on resume
            "rng_state_torch": torch.random.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate(),
        }

        if is_best:
            path = self.output_dir / "best_checkpoint.pth"
        else:
            path = self.output_dir / f"final_checkpoint_epoch{epoch}.pth"

        # Atomic save: write to .tmp then rename to prevent corruption
        # if the job is killed mid-write (common on HPC clusters)
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(path)

        if is_best:
            logger.info(f"Best checkpoint saved: {path} (dice_mean={self.best_val_dice:.4f})")
            try:
                mlflow.log_artifact(str(path))
            except Exception:
                pass
        else:
            logger.info(f"Final checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training state from a checkpoint for resumption.

        Restores model weights, optimizer, scheduler, scaler, and
        early stopping state so training can continue seamlessly.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.

        Returns:
            The epoch to resume from (checkpoint epoch + 1).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.best_val_dice = checkpoint.get("val_dice_mean", 0.0)
        self.best_val_metrics = {
            "dice_ET": checkpoint.get("val_dice_ET", 0.0),
            "dice_TC": checkpoint.get("val_dice_TC", 0.0),
            "dice_WT": checkpoint.get("val_dice_WT", 0.0),
            "dice_mean": checkpoint.get("val_dice_mean", 0.0),
        }

        resume_epoch = checkpoint["epoch"] + 1
        self.current_epoch = checkpoint["epoch"]

        if "early_stopping_state" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping_state"])
            logger.info(
                f"Early stopping state restored: counter={self.early_stopping.counter}, "
                f"best_value={self.early_stopping.best_value}, "
                f"triggered={self.early_stopping.triggered}"
            )
        else:
            logger.warning(
                "Checkpoint has no early_stopping_state (saved before this fix). "
                "Early stopping patience counter will reset."
            )

        # Restore random states for reproducibility
        if "rng_state_torch" in checkpoint:
            torch.random.set_rng_state(checkpoint["rng_state_torch"].cpu())
        if "rng_state_cuda" in checkpoint and torch.cuda.is_available():
            cuda_states = [state.cpu() if hasattr(state, 'cpu') else state for state in checkpoint["rng_state_cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        if "rng_state_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_state_numpy"])
        if "rng_state_python" in checkpoint:
            random.setstate(checkpoint["rng_state_python"])

        logger.info(
            f"Checkpoint loaded: epoch={checkpoint['epoch']}, "
            f"best_dice={self.best_val_dice:.4f}, resuming from epoch {resume_epoch}"
        )
        return resume_epoch

    def _handle_oom(self, error: RuntimeError):
        """Handle out-of-memory error gracefully.

        Logs the error, saves partial checkpoint, and exits cleanly.

        Args:
            error: The RuntimeError from CUDA OOM.
        """
        logger.error(f"OOM at epoch {self.current_epoch}: {error}")
        torch.cuda.empty_cache()

        try:
            mlflow.log_param("training_failed", "OOM")
            mlflow.log_param("oom_epoch", self.current_epoch)
            mlflow.log_metric("oom_vram_peak_mb", self.vram_profiler.get_peak_mb())
        except Exception:
            pass

        # Save partial checkpoint
        try:
            path = self.output_dir / f"oom_checkpoint_epoch{self.current_epoch}.pth"
            torch.save({
                "epoch": self.current_epoch,
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "best_val_dice": self.best_val_dice,
                "error": str(error),
            }, path)
            logger.info(f"Partial checkpoint saved before OOM exit: {path}")
        except Exception as save_err:
            logger.error(f"Failed to save OOM checkpoint: {save_err}")

        console.print(
            f"[bold red]OOM Error at epoch {self.current_epoch}. "
            f"Peak VRAM: {self.vram_profiler.get_peak_mb():.0f} MB. "
            f"Exiting gracefully.[/bold red]"
        )

    def _should_stop_for_wall_time(self) -> bool:
        """Check if we are approaching the Slurm wall-time deadline.

        Returns True ~10 minutes before the configured limit so we can
        save a checkpoint and exit with code 0.
        """
        return time.monotonic() >= self._wall_time_deadline
