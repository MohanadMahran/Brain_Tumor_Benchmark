"""Stratified split generator for deduplicated BraTS training pool.

Generates train/val splits independently for each dataset year,
then merges them into final training and validation sets.
Writes CSV manifests for reproducibility.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Modality file patterns for BraTS naming conventions
MODALITY_PATTERNS = {
    "brats2021": {
        "t1": "_t1.nii.gz",
        "t1ce": "_t1ce.nii.gz",
        "t2": "_t2.nii.gz",
        "flair": "_flair.nii.gz",
        "seg": "_seg.nii.gz",
    },
    "brats2024": {
        "t1": "-t1n.nii",
        "t1ce": "-t1c.nii",
        "t2": "-t2w.nii",
        "flair": "-t2f.nii",
        "seg": "-seg.nii",
    },
}


def find_modality_files(case_dir: Path, dataset_origin: str) -> Dict[str, str]:
    """Find modality file paths for a given case directory.

    Args:
        case_dir: Path to the case directory.
        dataset_origin: Either 'brats2021' or 'brats2024'.

    Returns:
        Dict mapping modality names to file paths.
    """
    patterns = MODALITY_PATTERNS.get(dataset_origin)
    if patterns is None:
        # Fallback: try to find files by common patterns
        patterns = MODALITY_PATTERNS["brats2021"]

    result = {}
    case_name = case_dir.name

    for mod_key, suffix in patterns.items():
        # Try exact pattern first
        candidates = list(case_dir.glob(f"*{suffix}"))
        if not candidates:
            # Try alternate patterns
            alt_patterns = [
                f"{case_name}{suffix}",
                f"*{mod_key}*.nii.gz",
                f"*{mod_key}*.nii",
            ]
            for alt in alt_patterns:
                candidates = list(case_dir.glob(alt))
                if candidates:
                    break

        if candidates:
            result[mod_key] = str(sorted(candidates)[0])
        else:
            logger.warning(f"Missing {mod_key} for case {case_name}")

    return result


def generate_splits(
    valid_cases_2021: List[str],
    valid_cases_2024: List[str],
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate stratified train/val splits from deduplicated case lists.

    Splits each year independently (80/20), then merges.

    Args:
        valid_cases_2021: List of valid case directory paths from BraTS 2021.
        valid_cases_2024: List of valid case directory paths from BraTS 2024.
        output_dir: Directory to write manifests.
        seed: Random seed for reproducibility.
        train_ratio: Fraction of cases for training.

    Returns:
        Tuple of (train_manifest_df, val_manifest_df).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split BraTS 2021
    train_2021, val_2021 = train_test_split(
        valid_cases_2021,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Split BraTS 2024
    train_2024, val_2024 = train_test_split(
        valid_cases_2024,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )

    console.print(f"  BraTS 2021: {len(train_2021)} train, {len(val_2021)} val")
    console.print(f"  BraTS 2024: {len(train_2024)} train, {len(val_2024)} val")

    # Build manifest rows
    train_rows = []
    val_rows = []

    for case_path in train_2021:
        row = _build_manifest_row(case_path, "brats2021")
        if row is not None:
            train_rows.append(row)

    for case_path in train_2024:
        row = _build_manifest_row(case_path, "brats2024")
        if row is not None:
            train_rows.append(row)

    for case_path in val_2021:
        row = _build_manifest_row(case_path, "brats2021")
        if row is not None:
            val_rows.append(row)

    for case_path in val_2024:
        row = _build_manifest_row(case_path, "brats2024")
        if row is not None:
            val_rows.append(row)

    # Create DataFrames
    columns = ["case_id", "t1_path", "t1ce_path", "t2_path", "flair_path",
               "label_path", "dataset_origin"]
    train_df = pd.DataFrame(train_rows, columns=columns)
    val_df = pd.DataFrame(val_rows, columns=columns)

    # Write manifests
    train_manifest_path = output_path / "train_manifest.csv"
    val_manifest_path = output_path / "val_manifest.csv"
    train_df.to_csv(train_manifest_path, index=False)
    val_df.to_csv(val_manifest_path, index=False)

    console.print(f"\n  [green]Final train set: {len(train_df)} cases[/green]")
    console.print(f"  [green]Final val set: {len(val_df)} cases[/green]")
    console.print(f"  Manifests saved to: {output_path}")

    # Summary
    summary = {
        "train_total": len(train_df),
        "val_total": len(val_df),
        "train_brats2021": len(train_2021),
        "train_brats2024": len(train_2024),
        "val_brats2021": len(val_2021),
        "val_brats2024": len(val_2024),
    }
    summary_path = output_path / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return train_df, val_df


def _build_manifest_row(case_path: str, dataset_origin: str) -> list:
    """Build a manifest row for a single case.

    Args:
        case_path: Path to case directory.
        dataset_origin: 'brats2021' or 'brats2024'.

    Returns:
        List of [case_id, t1_path, t1ce_path, t2_path, flair_path, label_path, dataset_origin]
        or None if required files are missing.
    """
    case_dir = Path(case_path)
    case_id = case_dir.name

    files = find_modality_files(case_dir, dataset_origin)
    required = ["t1", "t1ce", "t2", "flair", "seg"]
    for req in required:
        if req not in files:
            logger.warning(f"Skipping {case_id}: missing {req}")
            return None

    return [
        case_id,
        files["t1"],
        files["t1ce"],
        files["t2"],
        files["flair"],
        files["seg"],
        dataset_origin,
    ]
