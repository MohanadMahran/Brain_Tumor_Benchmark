"""Forward pass and parameter count verification test.

Verifies:
    1. UNet3D forward pass: [1, 4, 128, 128, 128] -> [1, 3, 128, 128, 128]
    2. UNETR forward pass:  [1, 4, 128, 128, 128] -> [1, 3, 128, 128, 128]
    3. Both models have ~19M parameters (18M-22M range)
    4. Parameter counts are within 10% of each other
    5. No BatchNorm in either model

Usage:
    python tests/test_forward_pass.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.table import Table

from models.unet3d import UNet3D
from models.unetr import UNETR
from models.param_counter import count_parameters

console = Console()


def test_unet3d():
    """Test UNet3D forward pass and parameter count."""
    console.print("\n[bold]=" * 60 + "[/bold]")
    console.print("[bold]Testing UNet3D[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")

    model = UNet3D(
        in_channels=4,
        out_channels=3,
        base_channels=60,
        groups=4,
        dropout_rates=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=0.0,
    )

    # Parameter count
    total_params = count_parameters(model)
    console.print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Verify parameter range
    assert 18_000_000 <= total_params <= 22_000_000, (
        f"UNet3D parameter count {total_params:,} not in 18M-22M range!"
    )
    console.print(f"  [green]✓ Parameter count in 18M-22M range[/green]")

    # Verify no BatchNorm
    has_bn = any(
        isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        for m in model.modules()
    )
    assert not has_bn, "UNet3D contains BatchNorm!"
    console.print(f"  [green]✓ No BatchNorm[/green]")

    # Forward pass
    model.eval()
    x = torch.randn(1, 4, 128, 128, 128)
    with torch.no_grad():
        y = model(x)

    console.print(f"  Input:  {x.shape}")
    console.print(f"  Output: {y.shape}")

    assert x.shape == torch.Size([1, 4, 128, 128, 128]), f"Bad input shape: {x.shape}"
    assert y.shape == torch.Size([1, 3, 128, 128, 128]), f"Bad output shape: {y.shape}"
    console.print(f"  [green]✓ Forward pass correct[/green]")

    # Verify output is finite
    assert torch.isfinite(y).all(), "Output contains NaN/Inf!"
    console.print(f"  [green]✓ Output is finite[/green]")

    return total_params


def test_unetr():
    """Test UNETR forward pass and parameter count."""
    console.print("\n[bold]=" * 60 + "[/bold]")
    console.print("[bold]Testing UNETR[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")

    model = UNETR(
        in_channels=4,
        out_channels=3,
        input_size=128,
        patch_size=16,
        embedding_dim=384,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
    )

    # Parameter count
    total_params = count_parameters(model)
    console.print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Verify parameter range
    assert 18_000_000 <= total_params <= 22_000_000, (
        f"UNETR parameter count {total_params:,} not in 18M-22M range!"
    )
    console.print(f"  [green]✓ Parameter count in 18M-22M range[/green]")

    # Verify no BatchNorm
    has_bn = any(
        isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        for m in model.modules()
    )
    assert not has_bn, "UNETR contains BatchNorm!"
    console.print(f"  [green]✓ No BatchNorm[/green]")

    # Forward pass
    model.eval()
    x = torch.randn(1, 4, 128, 128, 128)
    with torch.no_grad():
        y = model(x)

    console.print(f"  Input:  {x.shape}")
    console.print(f"  Output: {y.shape}")

    assert x.shape == torch.Size([1, 4, 128, 128, 128]), f"Bad input shape: {x.shape}"
    assert y.shape == torch.Size([1, 3, 128, 128, 128]), f"Bad output shape: {y.shape}"
    console.print(f"  [green]✓ Forward pass correct[/green]")

    # Verify output is finite
    assert torch.isfinite(y).all(), "Output contains NaN/Inf!"
    console.print(f"  [green]✓ Output is finite[/green]")

    return total_params


def test_parameter_parity(unet_params: int, unetr_params: int):
    """Verify parameter counts are within 10% of each other."""
    console.print("\n[bold]=" * 60 + "[/bold]")
    console.print("[bold]Parameter Parity Check[/bold]")
    console.print("[bold]=" * 60 + "[/bold]")

    max_params = max(unet_params, unetr_params)
    min_params = min(unet_params, unetr_params)
    relative_diff = (max_params - min_params) / max_params

    table = Table(title="Parameter Count Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Millions", justify="right", style="green")
    table.add_row("UNet3D", f"{unet_params:,}", f"{unet_params / 1e6:.2f}M")
    table.add_row("UNETR", f"{unetr_params:,}", f"{unetr_params / 1e6:.2f}M")
    table.add_row("Difference", f"{abs(unet_params - unetr_params):,}", f"{relative_diff*100:.2f}%")
    console.print(table)

    assert relative_diff <= 0.10, (
        f"Parameter count mismatch! UNet3D={unet_params:,} vs "
        f"UNETR={unetr_params:,} (diff={relative_diff*100:.2f}% > 10%)"
    )
    console.print(f"  [green]✓ Parameter parity within 10%[/green]")


def main():
    console.print("\n[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold] Forward Pass & Parameter Verification[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")

    try:
        unet_params = test_unet3d()
        unetr_params = test_unetr()
        test_parameter_parity(unet_params, unetr_params)

        console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
        console.print("[bold green] ALL TESTS PASSED ✓[/bold green]")
        console.print("[bold green]" + "=" * 60 + "[/bold green]")
        return 0

    except AssertionError as e:
        console.print(f"\n[bold red]TEST FAILED: {e}[/bold red]")
        return 1

    except Exception as e:
        console.print(f"\n[bold red]UNEXPECTED ERROR: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
