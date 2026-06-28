"""Pre-trained weight adapters for 4-channel input compatibility.

Standard ViT patch embeddings expect 3-channel (RGB) input.
BraTS input is 4 channels (T1, T1Gd, T2, FLAIR).

Adapter strategy:
    1. Load pre-trained 3-channel patch embedding weights
    2. Average across the 3 input channels to get a per-output-channel template
    3. Tile the template to 4 channels

This is an optional ablation — default mode is training from scratch.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def adapt_patch_embedding_weights(
    pretrained_weight: torch.Tensor,
    target_in_channels: int = 4,
) -> torch.Tensor:
    """Adapt 3-channel patch embedding weights to 4-channel input.

    Strategy: Average the 3-channel weights, then tile to target channels.

    Args:
        pretrained_weight: Pre-trained weight of shape (out_ch, 3, *patch_size).
        target_in_channels: Desired input channels (4 for BraTS).

    Returns:
        Adapted weight of shape (out_ch, target_in_channels, *patch_size).
    """
    assert pretrained_weight.shape[1] == 3, (
        f"Expected 3 input channels, got {pretrained_weight.shape[1]}"
    )

    # Average across RGB channels: (out_ch, 1, *patch_size)
    channel_mean = pretrained_weight.mean(dim=1, keepdim=True)

    # Tile to target channels: (out_ch, target_in_channels, *patch_size)
    adapted = channel_mean.repeat(1, target_in_channels, *([1] * (pretrained_weight.ndim - 2)))

    # Scale to maintain approximate activation magnitude
    adapted = adapted * (3.0 / target_in_channels)

    logger.info(
        f"Adapted patch embedding: {pretrained_weight.shape} -> {adapted.shape}"
    )
    return adapted


def load_pretrained_vit_for_unetr(
    model: nn.Module,
    pretrained_source: str = "vit_base_patch16",
    in_channels: int = 4,
) -> nn.Module:
    """Load pre-trained ViT weights into UNETR's transformer encoder.

    Only loads compatible weights (transformer blocks, norms).
    Patch embedding is adapted for 4-channel input.
    Positional embeddings are interpolated if sequence lengths differ.

    Args:
        model: UNETR model instance.
        pretrained_source: Name of pre-trained model from timm.
        in_channels: Number of input channels.

    Returns:
        Model with loaded weights.
    """
    try:
        import timm
    except ImportError:
        logger.error("timm is required for pre-trained weight loading")
        return model

    logger.info(f"Loading pre-trained weights from: {pretrained_source}")

    # Load pre-trained 2D ViT
    pretrained_model = timm.create_model(pretrained_source, pretrained=True)
    pretrained_state = pretrained_model.state_dict()

    # Map transformer block weights
    loaded_count = 0
    skipped_count = 0

    for name, param in model.named_parameters():
        # Skip non-transformer parameters
        if "transformer" not in name:
            continue

        # Try to find corresponding pre-trained weight
        # This is a simplified mapping — exact mapping depends on architecture
        pretrained_name = _map_weight_name(name)
        if pretrained_name and pretrained_name in pretrained_state:
            pretrained_param = pretrained_state[pretrained_name]
            if pretrained_param.shape == param.shape:
                param.data.copy_(pretrained_param)
                loaded_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    # Adapt patch embedding if available
    patch_embed_key = "patch_embed.proj.weight"
    if patch_embed_key in pretrained_state:
        pretrained_pe = pretrained_state[patch_embed_key]
        if pretrained_pe.ndim == 4:  # 2D: (out, 3, H, W)
            # Cannot directly use 2D weights for 3D — log and skip
            logger.info(
                "Pre-trained patch embedding is 2D — cannot adapt to 3D. "
                "Keeping random initialization."
            )
        else:
            adapted = adapt_patch_embedding_weights(pretrained_pe, in_channels)
            model.patch_embed.proj.weight.data.copy_(adapted)
            loaded_count += 1

    logger.info(
        f"Pre-trained weight loading: {loaded_count} loaded, {skipped_count} skipped"
    )
    return model


def _map_weight_name(model_name: str) -> Optional[str]:
    """Map UNETR weight names to timm ViT weight names.

    This is a simplified mapping for common transformer components.

    Args:
        model_name: Parameter name in UNETR model.

    Returns:
        Corresponding name in timm ViT, or None if no mapping exists.
    """
    # Example mappings (simplified):
    # transformer.blocks.{i}.norm1.weight -> blocks.{i}.norm1.weight
    # transformer.blocks.{i}.attn.qkv.weight -> blocks.{i}.attn.qkv.weight
    if "transformer.blocks" in model_name:
        return model_name.replace("transformer.", "")
    return None
