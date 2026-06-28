"""Parameter counting and verification utilities.

Ensures both models have approximately the same number of parameters
(within 5% of each other) as required for fair comparison.
"""

import logging
from typing import Dict

import torch.nn as nn
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        only_trainable: If True, count only parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module) -> Dict[str, int]:
    """Count parameters grouped by top-level module.

    Args:
        model: PyTorch model.

    Returns:
        Dict mapping module name to parameter count.
    """
    counts = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts[name] = count
    return counts


def verify_parameter_parity(
    model_a: nn.Module,
    model_b: nn.Module,
    model_a_name: str = "UNet3D",
    model_b_name: str = "UNETR",
    tolerance: float = 0.05,
) -> bool:
    """Verify that two models have similar parameter counts.

    Raises AssertionError if they differ by more than the tolerance.

    Args:
        model_a: First model.
        model_b: Second model.
        model_a_name: Display name for first model.
        model_b_name: Display name for second model.
        tolerance: Maximum allowed relative difference (0.05 = 5%).

    Returns:
        True if within tolerance.

    Raises:
        AssertionError: If parameter counts differ by more than tolerance.
    """
    params_a = count_parameters(model_a)
    params_b = count_parameters(model_b)

    # Relative difference
    max_params = max(params_a, params_b)
    min_params = min(params_a, params_b)
    relative_diff = (max_params - min_params) / max_params

    # Display comparison table
    table = Table(title="Parameter Count Verification")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Millions", justify="right", style="green")
    table.add_row(model_a_name, f"{params_a:,}", f"{params_a / 1e6:.2f}M")
    table.add_row(model_b_name, f"{params_b:,}", f"{params_b / 1e6:.2f}M")
    table.add_row("Difference", f"{abs(params_a - params_b):,}", f"{relative_diff*100:.2f}%")
    console.print(table)

    # Assert within tolerance
    assert relative_diff <= tolerance, (
        f"Parameter count mismatch! {model_a_name}={params_a:,} vs "
        f"{model_b_name}={params_b:,} (diff={relative_diff*100:.2f}% > {tolerance*100:.1f}%)"
    )

    logger.info(
        f"Parameter parity verified: {model_a_name}={params_a:,} vs "
        f"{model_b_name}={params_b:,} (diff={relative_diff*100:.2f}%)"
    )
    return True


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print detailed parameter breakdown by module.

    Args:
        model: PyTorch model.
        model_name: Display name.
    """
    total = count_parameters(model)
    by_module = count_parameters_by_module(model)

    table = Table(title=f"{model_name} — Parameter Breakdown")
    table.add_column("Module", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("% of Total", justify="right")

    for name, count in sorted(by_module.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        table.add_row(name, f"{count:,}", f"{pct:.1f}%")

    table.add_row("TOTAL", f"{total:,}", "100.0%", style="bold")
    console.print(table)

    # Check for BatchNorm (should not exist)
    has_batchnorm = any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for m in model.modules()
    )
    if has_batchnorm:
        console.print("[bold red]ERROR: BatchNorm detected! This is not allowed.[/bold red]")
        raise AssertionError(f"{model_name} contains BatchNorm layers!")
    else:
        console.print(f"[green]✓ No BatchNorm in {model_name}[/green]")
