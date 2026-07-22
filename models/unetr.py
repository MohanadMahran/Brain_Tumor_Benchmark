"""UNETR implementation for brain tumor segmentation.

Architecture:
    - Patch embedding: 16³ non-overlapping patches (dynamic token count)
    - Transformer: 6 layers, 8 heads, embedding_dim=384
    - Pre-norm with LayerNorm
    - Gradient checkpointing on all transformer layers
    - CNN decoder with skip connections from layers 3 and 6
    - Stochastic depth (drop path) regularization
    - NO BatchNorm — LayerNorm exclusively
    - Dynamic positional embedding interpolation for variable input sizes

Target: ~19M parameters.
"""

import logging
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Drop path (stochastic depth) regularization."""

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
        return x / keep_prob * random_tensor


class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding for volumetric data.

    Divides input volume into non-overlapping patches and projects
    each flattened patch to the embedding dimension.

    Positional embeddings are allocated at init for the given input_size,
    but dynamically interpolated in forward() if the actual input produces
    a different number of tokens (e.g., input_size=128 with patch_size=16
    produces 8^3=512 tokens).

    Args:
        in_channels: Input channels (4 for BraTS).
        patch_size: Patch dimensions (e.g., 16).
        embedding_dim: Token embedding dimension.
        input_size: Input spatial dimensions used to initialize pos_embed.
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 16,
        embedding_dim: int = 256,
        input_size: int = 96,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.init_grid_size = input_size // patch_size
        self.init_num_patches = self.init_grid_size ** 3

        # Linear projection via Conv3d with kernel_size = stride = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable absolute positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.init_num_patches, embedding_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _interpolate_pos_embed(
        self, pos_embed: torch.Tensor, grid_size: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings to match actual grid size.

        Reshapes the 1D token sequence to a 3D grid, applies trilinear
        interpolation, then flattens back.

        Args:
            pos_embed: Shape (1, init_num_patches, embedding_dim).
            grid_size: Actual grid size (tokens per spatial dimension).

        Returns:
            Interpolated pos_embed of shape (1, grid_size^3, embedding_dim).
        """
        g0 = self.init_grid_size
        # (1, g0^3, D) -> (1, D, g0, g0, g0)
        pos_embed = pos_embed.transpose(1, 2).reshape(
            1, self.embedding_dim, g0, g0, g0
        )
        # Interpolate to new grid size
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_size, grid_size, grid_size),
            mode="trilinear",
            align_corners=False,
        )
        # (1, D, g, g, g) -> (1, g^3, D)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert volume to sequence of patch embeddings.

        Args:
            x: Input of shape (B, C, D, H, W).

        Returns:
            Token sequence of shape (B, num_patches, embedding_dim).
        """
        # (B, C, D, H, W) -> (B, embed_dim, D', H', W')
        x = self.proj(x)
        B, D_emb, gD, gH, gW = x.shape
        actual_grid_size = gD  # assumes cubic input: gD == gH == gW

        # (B, embed_dim, D', H', W') -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # Interpolate positional embeddings if token count changed
        if actual_grid_size != self.init_grid_size:
            pos_embed = self._interpolate_pos_embed(self.pos_embed, actual_grid_size)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm.

    Args:
        embedding_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, embedding_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Token sequence of shape (B, N, D).

        Returns:
            Attended tokens of shape (B, N, D).
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm architecture.

    Args:
        embedding_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embedding_dim)
        mlp_hidden = embedding_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm residual connections.

        Args:
            x: Token sequence of shape (B, N, D).

        Returns:
            Processed tokens of shape (B, N, D).
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks with gradient checkpointing.

    Args:
        embedding_dim: Token embedding dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
        drop_path_rate: Maximum stochastic depth rate.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Linearly increasing drop path rates
        dpr = [drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        extraction_layers: List[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with intermediate feature extraction.

        Args:
            x: Token sequence of shape (B, N, D).
            extraction_layers: Layer indices to extract features from (1-indexed).

        Returns:
            Tuple of (final_output, list_of_intermediate_features).
        """
        if extraction_layers is None:
            extraction_layers = [3, 6]

        intermediate_features = []

        for i, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            # Extract features at specified layers (1-indexed)
            if (i + 1) in extraction_layers:
                intermediate_features.append(x)

        x = self.norm(x)
        return x, intermediate_features


class CNNDecoder(nn.Module):
    """3D CNN decoder for UNETR that upsamples token maps to output resolution.

    Reshapes token sequences back to spatial feature maps and progressively
    upsamples using transposed convolutions with skip connections.

    Args:
        embedding_dim: Transformer embedding dimension.
        out_channels: Number of output channels.
        grid_size: Spatial grid size of tokens (e.g., 6 for 96/16).
        base_decoder_channels: Base decoder channel width.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        in_channels: int = 4,
        out_channels: int = 3,
        grid_size: int = 6,
        base_decoder_channels: int = 32,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim

        # Initial projection from input (for first skip connection)
        self.input_proj = nn.Sequential(
            nn.Conv3d(in_channels, base_decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels),
            nn.LeakyReLU(inplace=True),
        )

        # Project intermediate features (layer 3)
        self.skip_proj_mid = nn.Sequential(
            nn.Conv3d(embedding_dim, base_decoder_channels * 4, kernel_size=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 4),
            nn.LeakyReLU(inplace=True),
        )

        # Project final features (layer 6)
        self.skip_proj_final = nn.Sequential(
            nn.Conv3d(embedding_dim, base_decoder_channels * 8, kernel_size=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 8),
            nn.LeakyReLU(inplace=True),
        )

        # Upsampling path
        # From bottleneck (256 channels at 6x6x6) up to 96x96x96

        # Level 3: 6x6x6 -> 12x12x12
        self.up3 = nn.ConvTranspose3d(
            base_decoder_channels * 8, base_decoder_channels * 4,
            kernel_size=2, stride=2,
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_decoder_channels * 8, base_decoder_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_decoder_channels * 4, base_decoder_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 4),
            nn.LeakyReLU(inplace=True),
        )

        # Level 2: 12x12x12 -> 24x24x24
        self.up2 = nn.ConvTranspose3d(
            base_decoder_channels * 4, base_decoder_channels * 2,
            kernel_size=2, stride=2,
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_decoder_channels * 2, base_decoder_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_decoder_channels * 2, base_decoder_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels * 2),
            nn.LeakyReLU(inplace=True),
        )

        # Level 1: 24x24x24 -> 48x48x48
        self.up1 = nn.ConvTranspose3d(
            base_decoder_channels * 2, base_decoder_channels,
            kernel_size=2, stride=2,
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_decoder_channels, base_decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_decoder_channels, base_decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels),
            nn.LeakyReLU(inplace=True),
        )

        # Level 0: 48x48x48 -> 96x96x96
        self.up0 = nn.ConvTranspose3d(
            base_decoder_channels, base_decoder_channels,
            kernel_size=2, stride=2,
        )
        self.dec0 = nn.Sequential(
            nn.Conv3d(base_decoder_channels * 2, base_decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_decoder_channels, base_decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_decoder_channels),
            nn.LeakyReLU(inplace=True),
        )

        # Final output
        self.output_conv = nn.Conv3d(base_decoder_channels, out_channels, kernel_size=1)

    def tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape token sequence to 3D spatial feature map.

        Args:
            tokens: Shape (B, N, D) where N = grid_size^3.

        Returns:
            Spatial map of shape (B, D, grid_size, grid_size, grid_size).
        """
        B, N, D = tokens.shape
        g = self.grid_size
        x = tokens.transpose(1, 2)  # (B, D, N)
        x = x.reshape(B, D, g, g, g)
        return x

    def forward(
        self,
        original_input: torch.Tensor,
        final_tokens: torch.Tensor,
        intermediate_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Decode transformer features to segmentation output.

        Args:
            original_input: Original input volume (B, 4, 96, 96, 96).
            final_tokens: Final transformer output tokens (B, N, D).
            intermediate_features: List of intermediate token sequences.

        Returns:
            Logits of shape (B, out_channels, 96, 96, 96).
        """
        # Reshape tokens to spatial maps (at 6x6x6 resolution)
        feat_final = self.tokens_to_spatial(final_tokens)  # (B, 256, 6, 6, 6)
        feat_mid = None
        if len(intermediate_features) > 0:
            feat_mid = self.tokens_to_spatial(intermediate_features[0])  # (B, 256, 6, 6, 6)

        # Project features
        x = self.skip_proj_final(feat_final)  # (B, 256, 6, 6, 6)

        # Upsample level 3: 6 -> 12
        x = self.up3(x)  # (B, 128, 12, 12, 12)
        if feat_mid is not None:
            skip_mid = self.skip_proj_mid(feat_mid)  # (B, 128, 6, 6, 6)
            skip_mid = F.interpolate(skip_mid, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip_mid], dim=1)  # (B, 256, 12, 12, 12)
        else:
            x = torch.cat([x, x], dim=1)  # Fallback
        x = self.dec3(x)  # (B, 128, 12, 12, 12)

        # Upsample level 2: 12 -> 24
        x = self.up2(x)  # (B, 64, 24, 24, 24)
        x = self.dec2(x)  # (B, 64, 24, 24, 24)

        # Upsample level 1: 24 -> 48
        x = self.up1(x)  # (B, 32, 48, 48, 48)
        x = self.dec1(x)  # (B, 32, 48, 48, 48)

        # Upsample level 0: 48 -> 96
        x = self.up0(x)  # (B, 32, 96, 96, 96)

        # Skip from original input
        input_feat = self.input_proj(original_input)  # (B, 32, 96, 96, 96)
        x = torch.cat([x, input_feat], dim=1)  # (B, 64, 96, 96, 96)
        x = self.dec0(x)  # (B, 32, 96, 96, 96)

        # Output
        logits = self.output_conv(x)  # (B, 3, 96, 96, 96)
        return logits


class UNETR(nn.Module):
    """UNETR for 3D brain tumor segmentation.

    Combines a Vision Transformer encoder with a CNN decoder.
    Uses gradient checkpointing to reduce VRAM usage.
    NO BatchNorm — LayerNorm exclusively.

    Args:
        in_channels: Input channels (4).
        out_channels: Output channels (3).
        input_size: Spatial dimension of input (128).
        patch_size: Token patch size (16).
        embedding_dim: Transformer embedding dimension (384).
        num_layers: Number of transformer blocks (6).
        num_heads: Number of attention heads (8).
        mlp_ratio: MLP expansion ratio (4).
        dropout: Dropout rate (0.0).
        drop_path_rate: Stochastic depth rate (0.0).
        use_checkpoint: Use gradient checkpointing (False).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        input_size: int = 128,
        patch_size: int = 16,
        embedding_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        grid_size = input_size // patch_size

        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            input_size=input_size,
        )

        self.transformer = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

        self.decoder = CNNDecoder(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            grid_size=grid_size,
            base_decoder_channels=32,
        )

        # Feature extraction layers (1-indexed): mid-layer and final layer
        self.extraction_layers = [num_layers // 2, num_layers]

        # Initialize weights
        self._init_weights()

        logger.info(
            f"UNETR initialized: embed_dim={embedding_dim}, layers={num_layers}, "
            f"heads={num_heads}, patches={grid_size}^3={grid_size**3}, "
            f"extraction_layers={self.extraction_layers}"
        )

    def _init_weights(self):
        """Initialize weights appropriately for each component."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input volume of shape (B, 4, 96, 96, 96).

        Returns:
            Logits of shape (B, 3, 96, 96, 96).
        """
        original_input = x

        # Patch embedding
        tokens = self.patch_embed(x)  # (B, 216, 256)

        # Transformer encoding with intermediate extraction
        final_tokens, intermediate = self.transformer(tokens, self.extraction_layers)

        # CNN decoding
        logits = self.decoder(original_input, final_tokens, intermediate)
        return logits

    def forward_with_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid for inference."""
        return torch.sigmoid(self.forward(x))
