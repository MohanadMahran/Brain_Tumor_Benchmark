"""Volumetric segmentation metrics: Dice coefficient and HD95.

Implements from scratch for 3D evaluation:
    - Dice coefficient per region (ET, TC, WT)
    - 95th percentile Hausdorff Distance (HD95) per region
    - Edge case handling for empty predictions/targets
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# Penalty for empty predictions — diagonal of 240x240x155 volume (mm)
HD95_PENALTY = np.sqrt(240**2 + 240**2 + 155**2)  # ~373.13 mm

REGION_NAMES = ["ET", "TC", "WT"]


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient between binary prediction and target.

    Dice = 2 * |pred ∩ target| / (|pred| + |target| + epsilon)

    Args:
        pred: Binary prediction array.
        target: Binary ground truth array.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum + target_sum == 0:
        # Both empty — perfect agreement
        return 1.0

    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
    return float(dice)


def compute_surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compute symmetric surface distances between prediction and target.

    Extracts surfaces using morphological erosion, then computes distances
    from each surface point to the nearest point on the other surface.

    Args:
        pred: Binary prediction array.
        target: Binary ground truth array.
        voxel_spacing: Voxel dimensions in mm.

    Returns:
        Array of all symmetric surface distances.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Extract surface points using erosion
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    pred_eroded = ndimage.binary_erosion(pred, structure=struct)
    pred_surface = pred & ~pred_eroded
    target_eroded = ndimage.binary_erosion(target, structure=struct)
    target_surface = target & ~target_eroded

    # If either surface is empty, return empty
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([])

    # Compute distance transform from target surface
    # (distance of each voxel to nearest target surface point)
    target_dt = ndimage.distance_transform_edt(~target_surface, sampling=voxel_spacing)
    pred_dt = ndimage.distance_transform_edt(~pred_surface, sampling=voxel_spacing)

    # Surface distances: from pred surface to nearest target surface point
    pred_to_target = target_dt[pred_surface]
    target_to_pred = pred_dt[target_surface]

    # Symmetric surface distances
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    return all_distances


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute 95th percentile Hausdorff Distance.

    Edge cases:
        - If both pred and target are empty: return 0.0 (perfect agreement)
        - If pred is empty but target is not (or vice versa): return penalty

    Args:
        pred: Binary prediction array.
        target: Binary ground truth array.
        voxel_spacing: Voxel dimensions in mm.

    Returns:
        HD95 in mm.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_empty = pred.sum() == 0
    target_empty = target.sum() == 0

    # Edge cases
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return HD95_PENALTY

    distances = compute_surface_distances(pred, target, voxel_spacing)
    if len(distances) == 0:
        return HD95_PENALTY

    return float(np.percentile(distances, 95))


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute all metrics for a single case.

    Expects pred and target with 3 channels: [ET, TC, WT].

    Args:
        pred: Binary prediction of shape (3, D, H, W).
        target: Binary ground truth of shape (3, D, H, W).
        voxel_spacing: Voxel dimensions in mm.

    Returns:
        Dict with keys: dice_ET, dice_TC, dice_WT, dice_mean,
                        hd95_ET, hd95_TC, hd95_WT, hd95_mean
    """
    assert pred.shape[0] == 3, f"Expected 3 channels, got {pred.shape[0]}"
    assert target.shape[0] == 3, f"Expected 3 channels, got {target.shape[0]}"

    metrics = {}
    dice_scores = []
    hd95_scores = []

    for i, region in enumerate(REGION_NAMES):
        pred_region = pred[i]
        target_region = target[i]

        dice = compute_dice(pred_region, target_region)
        hd95 = compute_hd95(pred_region, target_region, voxel_spacing)

        metrics[f"dice_{region}"] = dice
        metrics[f"hd95_{region}"] = hd95

        dice_scores.append(dice)
        hd95_scores.append(hd95)

    metrics["dice_mean"] = float(np.mean(dice_scores))
    metrics["hd95_mean"] = float(np.mean(hd95_scores))

    return metrics
