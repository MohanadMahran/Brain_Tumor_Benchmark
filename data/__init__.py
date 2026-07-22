"""Data pipeline for BraTS brain tumor segmentation benchmark.

Modules:
    deduplication: MD5-based cross-year deduplication
    dataset: Unified BraTSDataset class
    preprocessing: Per-modality Z-score normalization
    augmentation: MONAI-based training augmentation pipeline
    splits: Stratified train/val split generation
"""

from data.dataset import BraTSDataset
from data.preprocessing import preprocess_volume
from data.augmentation import get_train_transforms, get_val_transforms, get_overfit_transforms
from data.splits import generate_splits
from data.deduplication import deduplicate_datasets

__all__ = [
    "BraTSDataset",
    "preprocess_volume",
    "get_train_transforms",
    "get_val_transforms",
    "get_overfit_transforms",
    "generate_splits",
    "deduplicate_datasets",
]

