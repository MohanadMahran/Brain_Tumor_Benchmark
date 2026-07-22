"""3D U-Net implementation for brain tumor segmentation.

Architecture:
    - Encoder: 4 levels of double conv blocks (GroupNorm + LeakyReLU)
    - Channel progression: 4 -> 32 -> 64 -> 128 -> 256
    - Downsampling: MaxPool3d(2,2,2)
    - Decoder: Transposed conv upsampling + skip concatenation
    - Output: 3 sigmoid channels (ET, TC, WT)
    - NO BatchNorm — uses GroupNorm(groups=8) exclusively
    - Stochastic depth (drop path) for regularization

Target: ~19M parameters.
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Drop path (stochastic depth) for regularization.

    Randomly drops entire residual branches during training.

    Args:
        drop_prob: Probability of dropping the path.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class ConvBlock(nn.Module):
    """Double 3D convolution block with GroupNorm and LeakyReLU.

    Each block: Conv3d -> GroupNorm -> LeakyReLU -> Conv3d -> GroupNorm -> LeakyReLU
    With optional residual connection and drop path.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        groups: Number of groups for GroupNorm.
        dropout: Dropout probability.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Residual connection (1x1 conv if channels change)
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.gn2(out)

        # Apply drop path to the residual branch
        out = self.drop_path(out) + residual
        out = self.act(out)
        return out


class Encoder(nn.Module):
    """4-level 3D U-Net encoder.

    Args:
        in_channels: Input channels (4 for BraTS).
        base_channels: Base channel count.
        groups: GroupNorm groups.
        dropout_rates: Per-level dropout rates.
        drop_path_rate: Maximum drop path rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        groups: int = 8,
        dropout_rates: List[float] = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.1, 0.2, 0.2, 0.2]

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Linearly increasing drop path rates
        dpr = [drop_path_rate * i / 3 for i in range(4)]

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Level 0
        self.blocks.append(
            ConvBlock(in_channels, channels[0], groups, dropout_rates[0], dpr[0])
        )

        # Levels 1-3
        for i in range(1, 4):
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.blocks.append(
                ConvBlock(channels[i - 1], channels[i], groups, dropout_rates[i], dpr[i])
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning skip connections at each level.

        Returns:
            List of feature maps [level0, level1, level2, level3(bottleneck)].
        """
        features = []
        out = self.blocks[0](x)
        features.append(out)

        for i in range(1, 4):
            out = self.pools[i - 1](out)
            out = self.blocks[i](out)
            features.append(out)

        return features


class Decoder(nn.Module):
    """3D U-Net decoder with transposed convolution upsampling.

    Args:
        base_channels: Base channel count (same as encoder).
        groups: GroupNorm groups.
    """

    def __init__(self, base_channels: int = 32, groups: int = 8):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Upsampling blocks (from bottleneck up)
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(2, -1, -1):  # levels 2, 1, 0
            self.up_convs.append(
                nn.ConvTranspose3d(
                    channels[i + 1], channels[i], kernel_size=2, stride=2
                )
            )
            # After concatenation with skip: channels[i]*2 -> channels[i]
            self.conv_blocks.append(
                ConvBlock(channels[i] * 2, channels[i], groups)
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Decode from bottleneck using skip connections.

        Args:
            features: List [level0, level1, level2, bottleneck] from encoder.

        Returns:
            Decoded feature map at original resolution.
        """
        x = features[-1]  # Bottleneck

        for i, (up_conv, conv_block) in enumerate(zip(self.up_convs, self.conv_blocks)):
            x = up_conv(x)
            skip = features[-(i + 2)]  # Corresponding skip connection

            # Handle size mismatches from non-power-of-2 dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = conv_block(x)

        return x


class UNet3D(nn.Module):
    """3D U-Net for brain tumor segmentation.

    Architecture:
        - 4-level encoder with GroupNorm and LeakyReLU
        - Skip connections via concatenation
        - Transposed conv decoder
        - 3-channel sigmoid output (ET, TC, WT)
        - Stochastic depth regularization

    Target: ~19M parameters.

    Args:
        in_channels: Input channels (4 for T1, T1ce, T2, FLAIR).
        out_channels: Output channels (3 for ET, TC, WT).
        base_channels: Base channel width.
        groups: GroupNorm number of groups.
        dropout_rates: Per-encoder-level dropout rates.
        drop_path_rate: Maximum stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        groups: int = 8,
        dropout_rates: List[float] = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.1, 0.2, 0.2, 0.2]

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            groups=groups,
            dropout_rates=dropout_rates,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = Decoder(base_channels=base_channels, groups=groups)

        # Output head: 1x1x1 conv to desired channels
        self.output_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"UNet3D initialized: in={in_channels}, out={out_channels}, "
            f"base_ch={base_channels}, groups={groups}"
        )

    def _init_weights(self):
        """Initialize weights: Kaiming normal for conv, constant 1 for GroupNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 4, D, H, W).

        Returns:
            Logits tensor of shape (B, 3, D, H, W).
            Apply sigmoid externally for probability maps.
        """
        features = self.encoder(x)
        decoded = self.decoder(features)
        logits = self.output_conv(decoded)
        return logits

    def forward_with_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation for inference.

        Args:
            x: Input tensor of shape (B, 4, D, H, W).

        Returns:
            Probability tensor of shape (B, 3, D, H, W) in [0, 1].
        """
        return torch.sigmoid(self.forward(x))
