"""Per-modality preprocessing pipeline for BraTS NIfTI volumes.

Applies identical preprocessing to training, validation, and benchmark data:
1. Load NIfTI volume
2. Extract brain mask (volume > 0 for skull-stripped data)
3. Clip to [0.5th, 99.5th] percentile within brain mask
4. Z-score normalize using brain mask statistics
5. Zero-fill outside brain mask
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return the volume data as float32.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        3D numpy array of shape (H, W, D) as float32.
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """Apply Z-score normalization to a single modality volume.

    Steps:
        1. Compute brain mask (non-zero voxels in skull-stripped data)
        2. Clip intensities to [0.5th, 99.5th] percentile within brain mask
        3. Z-score normalize using brain mask mean and std
        4. Set non-brain voxels to 0.0

    Args:
        volume: 3D array of shape (H, W, D).

    Returns:
        Normalized 3D array of same shape.
    """
    # Extract brain mask
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        # Empty volume — return zeros
        logger.warning("Empty volume detected (no brain voxels). Returning zeros.")
        return np.zeros_like(volume)

    # Get brain-only values
    brain_values = volume[brain_mask]

    # Clip to [0.5th, 99.5th] percentile within brain mask
    p_low = np.percentile(brain_values, 0.5)
    p_high = np.percentile(brain_values, 99.5)
    clipped = np.clip(volume, p_low, p_high)

    # Z-score normalization using brain mask statistics
    brain_clipped = clipped[brain_mask]
    mean_brain = brain_clipped.mean()
    std_brain = brain_clipped.std()
    normalized = (clipped - mean_brain) / (std_brain + 1e-8)

    # Zero-fill outside brain mask
    normalized[~brain_mask] = 0.0

    return normalized


def preprocess_volume(
    modality_paths: Dict[str, str],
    label_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Preprocess a full multi-modal BraTS case.

    Loads all 4 modalities, normalizes each independently, and stacks them.
    Optionally loads and processes the segmentation label.

    Args:
        modality_paths: Dict with keys 't1', 't1ce', 't2', 'flair' mapping to file paths.
        label_path: Optional path to segmentation NIfTI file.

    Returns:
        Tuple of:
            - image: np.ndarray of shape (4, H, W, D), float32
            - label: np.ndarray of shape (3, H, W, D) or None if label_path is None.
              Channels are: [ET (label 4), TC (labels 1+4), WT (labels 1+2+4)]
    """
    modality_order = ["t1", "t1ce", "t2", "flair"]
    channels = []

    for mod in modality_order:
        if mod not in modality_paths:
            raise ValueError(f"Missing modality '{mod}' in provided paths.")
        vol = load_nifti(modality_paths[mod])
        normalized = normalize_modality(vol)
        channels.append(normalized)

    # Stack to (4, H, W, D)
    image = np.stack(channels, axis=0).astype(np.float32)

    # Process label if provided
    label = None
    if label_path is not None:
        seg = load_nifti(label_path).astype(np.int32)
        label = convert_labels_to_regions(seg)

    return image, label


def convert_labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """Convert BraTS label map to 3-channel binary region masks.

    BraTS label schema:
        0 = Background
        1 = NCR (Necrotic tumor core)
        2 = ED (Peritumoral edema)
        4 = ET (Enhancing tumor) (Note: label 3 is mapped to 4 for compatibility)

    Evaluation regions:
        ET = label 4
        TC = labels 1 + 4
        WT = labels 1 + 2 + 4

    Args:
        seg: Integer segmentation map of shape (H, W, D).

    Returns:
        Binary region masks of shape (3, H, W, D) for [ET, TC, WT].
    """
    # IMPORTANT: Remap label 3 → 4 for UPenn-GBM compatibility.
    # UPenn-GBM uses label 3 for enhancing tumor (ET), whereas the BraTS
    # competition standard uses label 4 for ET.  BraTS 2021, 2024, and SSA
    # datasets already use label 4 for ET and *never* contain label 3 in
    # their ground-truth segmentations (label values are {0, 1, 2, 4}), so
    # this np.where is a safe no-op for those datasets.
    # DO NOT REMOVE this line — it is required for correct UPenn-GBM evaluation.
    seg = np.where(seg == 3, 4, seg)

    et = (seg == 4).astype(np.float32)
    tc = np.logical_or(seg == 1, seg == 4).astype(np.float32)
    wt = np.logical_or(np.logical_or(seg == 1, seg == 2), seg == 4).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)
