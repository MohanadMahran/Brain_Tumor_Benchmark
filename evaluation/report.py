"""Report generation for benchmark comparison.

Generates CSV comparison tables and formatted summaries of benchmark results.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def generate_comparison_report(
    reports_dir: str,
    output_path: str,
) -> pd.DataFrame:
    """Generate final comparison report from all benchmark results.

    Reads per-model result CSVs from each benchmark and produces
    a unified comparison table.

    Args:
        reports_dir: Directory containing benchmark subdirectories.
        output_path: Path to write final comparison CSV.

    Returns:
        Comparison DataFrame.
    """
    reports_path = Path(reports_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # Scan benchmark directories
    for benchmark_dir in sorted(reports_path.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark_name = benchmark_dir.name

        # Load each model's results
        for results_file in sorted(benchmark_dir.glob("*_results.csv")):
            model_name = results_file.stem.replace("_results", "")
            try:
                df = pd.read_csv(results_file)

                # Compute aggregate metrics
                row = {
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "num_cases": len(df),
                }

                metric_cols = ["dice_ET", "dice_TC", "dice_WT", "dice_mean",
                               "hd95_ET", "hd95_TC", "hd95_WT", "hd95_mean"]
                for col in metric_cols:
                    if col in df.columns:
                        row[col] = df[col].mean()

                # Per-tumor-type for UPenn-GBM
                if "tumor_type" in df.columns:
                    for tt in ["GBM", "LGG"]:
                        subset = df[df["tumor_type"] == tt]
                        if len(subset) > 0:
                            row[f"dice_mean_{tt}"] = subset["dice_mean"].mean()

                results.append(row)
            except Exception as e:
                logger.warning(f"Failed to read {results_file}: {e}")

    if not results:
        logger.warning("No benchmark results found.")
        comparison_df = pd.DataFrame()
        comparison_df.to_csv(output_file, index=False)
        return comparison_df

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(output_file, index=False)

    # Print formatted table
    _print_comparison_table(comparison_df)

    console.print(f"\n[green]Report saved to: {output_file}[/green]")
    return comparison_df


def _print_comparison_table(df: pd.DataFrame):
    """Print a rich-formatted comparison table.

    Args:
        df: Comparison DataFrame.
    """
    table = Table(title="Brain Tumor Segmentation Benchmark Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Benchmark", style="magenta")
    table.add_column("Cases", justify="right")
    table.add_column("Dice ET", justify="right")
    table.add_column("Dice TC", justify="right")
    table.add_column("Dice WT", justify="right")
    table.add_column("Dice Mean", justify="right", style="bold green")
    table.add_column("HD95 Mean", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            str(row.get("model", "")),
            str(row.get("benchmark", "")),
            str(int(row.get("num_cases", 0))),
            f"{row.get('dice_ET', 0):.4f}",
            f"{row.get('dice_TC', 0):.4f}",
            f"{row.get('dice_WT', 0):.4f}",
            f"{row.get('dice_mean', 0):.4f}",
            f"{row.get('hd95_mean', 0):.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("--reports_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_comparison_report(args.reports_dir, args.output)
