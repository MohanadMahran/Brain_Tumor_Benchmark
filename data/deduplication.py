"""MD5-based cross-year deduplication for BraTS 2021 and BraTS 2024.

BraTS 2021 and BraTS 2024 have known patient overlap (estimated 200-300 cases).
This module identifies duplicates by computing MD5 hashes of concatenated
volume data and removes the older (2021) version when matches are found.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def compute_case_md5(case_dir: Path, modalities: List[str]) -> str:
    """Compute MD5 hash of all modality volumes concatenated for a given case.

    Args:
        case_dir: Path to the case directory containing NIfTI files.
        modalities: List of modality suffixes to include (e.g., ['t1', 't1ce', 't2', 'flair']).

    Returns:
        Hex digest of MD5 hash of concatenated volume data.
    """
    hasher = hashlib.md5()

    for mod in sorted(modalities):  # Sort for deterministic ordering
        # Find the modality file (handle different naming conventions)
        nifti_files = list(case_dir.glob(f"*{mod}*.nii.gz")) + \
                      list(case_dir.glob(f"*{mod}*.nii"))

        if not nifti_files:
            raise FileNotFoundError(
                f"No NIfTI file found for modality '{mod}' in {case_dir}"
            )

        nifti_path = sorted(nifti_files)[0]  # Take first match deterministically
        img = nib.load(str(nifti_path))
        data = img.get_fdata(dtype=np.float32)
        hasher.update(data.tobytes())

    return hasher.hexdigest()


def find_cases(dataset_dir: Path) -> List[Path]:
    """Find all case directories in a BraTS dataset directory.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Sorted list of case directory paths.
    """
    cases = []
    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith('.'):
            # Verify it contains NIfTI files
            nifti_files = list(entry.glob("*.nii.gz")) + list(entry.glob("*.nii"))
            if len(nifti_files) >= 4:  # At least 4 modalities
                cases.append(entry)
    return cases


def deduplicate_datasets(
    brats2021_dir: str,
    brats2024_dir: str,
    output_dir: str,
    modalities: List[str] = None,
) -> Dict:
    """Perform MD5-based deduplication between BraTS 2021 and BraTS 2024.

    When duplicates are found, the BraTS 2024 version is kept (newer annotations).

    Args:
        brats2021_dir: Path to BraTS 2021 dataset root.
        brats2024_dir: Path to BraTS 2024 dataset root.
        output_dir: Path to write deduplication report.
        modalities: List of modality suffixes. Defaults to ['t1', 't1ce', 't2', 'flair'].

    Returns:
        Dictionary with deduplication results including lists of valid cases.
    """
    if modalities is None:
        modalities = ["t1", "t1ce", "t2", "flair"]

    brats2021_path = Path(brats2021_dir)
    brats2024_path = Path(brats2024_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Starting cross-year deduplication...[/bold blue]")

    # Find all cases
    cases_2021 = find_cases(brats2021_path)
    cases_2024 = find_cases(brats2024_path)
    console.print(f"  BraTS 2021: {len(cases_2021)} cases found")
    console.print(f"  BraTS 2024: {len(cases_2024)} cases found")

    # Compute MD5 hashes for BraTS 2024 first (these take priority)
    hash_to_case_2024: Dict[str, Path] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Hashing BraTS 2024...", total=len(cases_2024))
        for case_dir in cases_2024:
            try:
                md5 = compute_case_md5(case_dir, modalities)
                hash_to_case_2024[md5] = case_dir
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Skipping {case_dir.name}: {e}")
            progress.advance(task)

    # Compute MD5 hashes for BraTS 2021 and check for duplicates
    duplicates_found = []
    valid_cases_2021 = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Hashing BraTS 2021...", total=len(cases_2021))
        for case_dir in cases_2021:
            try:
                md5 = compute_case_md5(case_dir, modalities)
                if md5 in hash_to_case_2024:
                    duplicates_found.append({
                        "brats2021_case": case_dir.name,
                        "brats2024_case": hash_to_case_2024[md5].name,
                        "md5": md5,
                    })
                else:
                    valid_cases_2021.append(case_dir)
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Skipping {case_dir.name}: {e}")
            progress.advance(task)

    valid_cases_2024 = list(hash_to_case_2024.values())

    # Warning if zero duplicates — likely hashing failure
    if len(duplicates_found) == 0:
        logger.warning(
            "ZERO duplicates found between BraTS 2021 and BraTS 2024. "
            "This likely indicates a hashing failure rather than genuine "
            "overlap-free datasets. Please verify data integrity."
        )
        console.print(
            "[bold yellow][WARNING] Zero duplicates found. "
            "Verify data integrity.[/bold yellow]"
        )

    # Build report
    report = {
        "total_2021": len(cases_2021),
        "total_2024": len(cases_2024),
        "duplicates_found": len(duplicates_found),
        "duplicates_removed_from_2021": len(duplicates_found),
        "valid_cases_2021": len(valid_cases_2021),
        "valid_cases_2024": len(valid_cases_2024),
        "final_pool_size": len(valid_cases_2021) + len(valid_cases_2024),
        "duplicate_details": duplicates_found,
    }

    # Write report
    report_path = output_path / "deduplication_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[bold green]Deduplication complete:[/bold green]")
    console.print(f"  Duplicates found: {len(duplicates_found)}")
    console.print(f"  Final pool size: {report['final_pool_size']}")
    console.print(f"  Report saved to: {report_path}")

    return {
        "report": report,
        "report_path": str(report_path),
        "valid_cases_2021": [str(p) for p in valid_cases_2021],
        "valid_cases_2024": [str(p) for p in valid_cases_2024],
    }
