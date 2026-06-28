"""Unified BraTSDataset class for training, validation, and benchmark data.

Handles lazy loading, preprocessing, and optional augmentation.
Works with CSV manifests produced by the splits module.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.preprocessing import preprocess_volume, load_nifti, normalize_modality

logger = logging.getLogger(__name__)


class BraTSDataset(Dataset):
    """Unified dataset class for BraTS brain tumor segmentation.

    Loads cases from a CSV manifest, applies preprocessing at load time,
    and optional MONAI transforms for augmentation.

    Args:
        manifest_path: Path to CSV manifest with columns:
            case_id, t1_path, t1ce_path, t2_path, flair_path, label_path, dataset_origin
        transform: Optional MONAI Compose transform pipeline.
        cache_data: Whether to cache preprocessed volumes in memory.
        include_label: Whether to load segmentation labels.
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        include_label: bool = True,
    ):
        self.manifest = pd.read_csv(manifest_path)
        self.transform = transform
        self.cache_data = cache_data
        self.include_label = include_label
        self._cache: Dict[int, Dict] = {}

        logger.info(
            f"BraTSDataset initialized: {len(self.manifest)} cases "
            f"from {manifest_path}"
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single case.

        Returns:
            Dict with keys:
                - 'image': tensor of shape (4, H, W, D)
                - 'label': tensor of shape (3, H, W, D) [if include_label]
                - 'case_id': string identifier
                - 'dataset_origin': 'brats2021' or 'brats2024'
        """
        if self.cache_data and idx in self._cache:
            data = self._cache[idx].copy()
        else:
            data = self._load_case(idx)
            if self.cache_data:
                self._cache[idx] = data.copy()

        # Apply transforms (MONAI expects dict with numpy arrays or tensors)
        if self.transform is not None:
            data = self.transform(data)

        # Ensure tensors
        if not isinstance(data["image"], torch.Tensor):
            data["image"] = torch.as_tensor(data["image"], dtype=torch.float32)
        if self.include_label and "label" in data:
            if not isinstance(data["label"], torch.Tensor):
                data["label"] = torch.as_tensor(data["label"], dtype=torch.float32)

        return data

    def _load_case(self, idx: int) -> Dict:
        """Load and preprocess a case from disk.

        Args:
            idx: Index into manifest.

        Returns:
            Dict with preprocessed numpy arrays.
        """
        row = self.manifest.iloc[idx]
        modality_paths = {
            "t1": row["t1_path"],
            "t1ce": row["t1ce_path"],
            "t2": row["t2_path"],
            "flair": row["flair_path"],
        }
        label_path = row["label_path"] if self.include_label else None
        image, label = preprocess_volume(modality_paths, label_path)

        data = {
            "image": image,
            "case_id": row["case_id"],
            "dataset_origin": row["dataset_origin"],
        }
        if label is not None:
            data["label"] = label

        return data

    def get_origin_indices(self, origin: str) -> list:
        """Get indices of cases from a specific dataset origin.

        Args:
            origin: 'brats2021' or 'brats2024'

        Returns:
            List of integer indices.
        """
        mask = self.manifest["dataset_origin"] == origin
        return self.manifest.index[mask].tolist()


class BenchmarkDataset(Dataset):
    """Dataset class for benchmark evaluation (TCGA, BraTS-SSA).

    Similar to BraTSDataset but handles different directory structures
    and includes metadata about tumor type for TCGA.

    Args:
        data_dir: Root directory of benchmark dataset.
        benchmark_type: 'tcga' or 'ssa'.
        transform: Optional transform pipeline.
    """

    def __init__(
        self,
        data_dir: str,
        benchmark_type: str = "tcga",
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.benchmark_type = benchmark_type
        self.transform = transform
        self.cases = self._discover_cases()

        logger.info(
            f"BenchmarkDataset ({benchmark_type}): {len(self.cases)} cases "
            f"from {data_dir}"
        )

    def _discover_cases(self) -> list:
        """Discover all valid cases in the benchmark directory."""
        cases = []
        for entry in sorted(self.data_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            nifti_files = list(entry.glob("*.nii.gz")) + list(entry.glob("*.nii"))
            if len(nifti_files) >= 4:
                # Determine tumor type for TCGA
                tumor_type = "unknown"
                if self.benchmark_type == "tcga":
                    name_lower = entry.name.lower()
                    if "gbm" in name_lower:
                        tumor_type = "GBM"
                    elif "lgg" in name_lower:
                        tumor_type = "LGG"
                cases.append({
                    "case_dir": entry,
                    "case_id": entry.name,
                    "tumor_type": tumor_type,
                })
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        """Load a benchmark case with full-volume preprocessing."""
        case_info = self.cases[idx]
        case_dir = case_info["case_dir"]

        # Find modality files
        modality_paths = self._find_modalities(case_dir)
        label_path = self._find_label(case_dir)
        image, label = preprocess_volume(modality_paths, label_path)

        data = {
            "image": torch.as_tensor(image, dtype=torch.float32),
            "case_id": case_info["case_id"],
            "tumor_type": case_info["tumor_type"],
        }
        if label is not None:
            data["label"] = torch.as_tensor(label, dtype=torch.float32)

        return data

    def _find_modalities(self, case_dir: Path) -> Dict[str, str]:
        """Find modality files using flexible pattern matching."""
        modalities = {}
        patterns = {
            "t1": ["*t1.nii*", "*_t1_*", "*-t1n*", "*t1.nii.gz"],
            "t1ce": ["*t1ce*", "*t1Gd*", "*t1c*", "*-t1c.*"],
            "t2": ["*t2.nii*", "*_t2_*", "*-t2w*", "*t2.nii.gz"],
            "flair": ["*flair*", "*-t2f*", "*FLAIR*"],
        }
        for mod, pats in patterns.items():
            for pat in pats:
                found = list(case_dir.glob(pat))
                # Exclude segmentation files
                found = [f for f in found if "seg" not in f.name.lower()]
                if found:
                    modalities[mod] = str(sorted(found)[0])
                    break
            if mod not in modalities:
                logger.warning(f"Missing {mod} in {case_dir.name}")

        return modalities

    def _find_label(self, case_dir: Path) -> Optional[str]:
        """Find segmentation label file."""
        patterns = ["*seg*", "*label*", "*mask*"]
        for pat in patterns:
            found = list(case_dir.glob(pat))
            if found:
                return str(sorted(found)[0])
        return None
