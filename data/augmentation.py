"""MONAI-based augmentation pipelines for training and validation.

Training pipeline applies spatial and intensity augmentations.
Validation pipeline applies only center cropping.
Both models receive the EXACT same augmented data from a single DataLoader.
"""

from typing import List, Tuple

from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureTyped,
    ToTensord,
)


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    keys: List[str] = None,
) -> Compose:
    """Build training augmentation pipeline.

    Applied only during training, never during validation or benchmarking.

    Pipeline:
        1. Pad to ensure minimum size >= patch_size
        2. Random spatial crop to patch_size
        3. Random flips (p=0.5, each axis independently)
        4. Random 90-degree rotations (p=0.5)
        5. Random intensity scaling (p=0.3)
        6. Random intensity shifting (p=0.3)
        7. Random Gaussian noise (p=0.2)
        8. Random Gaussian smoothing (p=0.2)

    Args:
        patch_size: Training patch dimensions.
        keys: Dictionary keys for image and label. Defaults to ["image", "label"].

    Returns:
        MONAI Compose transform pipeline.
    """
    if keys is None:
        keys = ["image", "label"]

    image_key = keys[0]

    transforms = [
        # Ensure minimum spatial size for cropping
        SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant"),
        # Random spatial crop — training patch size
        RandSpatialCropd(
            keys=keys,
            roi_size=patch_size,
            random_size=False,
        ),
        # Random flips — p=0.5 along each axis independently
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        # Random 90-degree rotation
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
        # Intensity augmentations — image only, not label
        RandScaleIntensityd(keys=[image_key], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=[image_key], offsets=0.1, prob=0.3),
        RandGaussianNoised(keys=[image_key], prob=0.2, mean=0.0, std=0.01),
        RandGaussianSmoothd(
            keys=[image_key],
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        # Ensure tensor type
        EnsureTyped(keys=keys, dtype="float32"),
    ]

    return Compose(transforms)


def get_val_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    keys: List[str] = None,
) -> Compose:
    """Build validation transform pipeline.

    Only center cropping — no random augmentation of any kind.

    Args:
        patch_size: Evaluation patch dimensions (center crop size).
        keys: Dictionary keys for image and label. Defaults to ["image", "label"].

    Returns:
        MONAI Compose transform pipeline.
    """
    if keys is None:
        keys = ["image", "label"]

    transforms = [
        # Pad to ensure minimum size
        SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant"),
        # Center crop only — no random operations
        CenterSpatialCropd(keys=keys, roi_size=patch_size),
        # Ensure tensor type
        EnsureTyped(keys=keys, dtype="float32"),
    ]

    return Compose(transforms)
