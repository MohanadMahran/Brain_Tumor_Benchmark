"""Training infrastructure for brain tumor segmentation benchmark.

Provides:
    - Unified Trainer class
    - DiceCE loss
    - Dice + HD95 metrics
    - Early stopping
    - VRAM profiling
"""

from training.trainer import Trainer
from training.losses import DiceCELoss
from training.metrics import compute_dice, compute_hd95, compute_all_metrics
from training.early_stopping import EarlyStopping
from training.vram_profiler import VRAMProfiler

__all__ = [
    "Trainer",
    "DiceCELoss",
    "compute_dice",
    "compute_hd95",
    "compute_all_metrics",
    "EarlyStopping",
    "VRAMProfiler",
]
