"""DiceCE loss implementation for brain tumor segmentation.

Combines Dice loss and Binary Cross-Entropy loss with equal weighting.
Applied per output channel (ET, TC, WT) independently, then averaged.
Supports label smoothing as an advanced regularization technique.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss.

    For each of the 3 output channels (ET, TC, WT):
        dice_loss = 1 - (2 * sum(pred * target) + smooth) / (sum(pred) + sum(target) + smooth)
        ce_loss = BCE(pred_logit, target)
        channel_loss = 0.5 * dice_loss + 0.5 * ce_loss

    Final loss = mean over 3 channels.

    Args:
        smooth: Smoothing constant for Dice denominator (prevents division by zero).
        dice_weight: Weight for Dice component.
        ce_weight: Weight for CE component.
        label_smoothing: Epsilon for label smoothing (0.0 = no smoothing).
    """

    def __init__(
        self,
        smooth: float = 1.0,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute DiceCE loss.

        Args:
            logits: Raw model output of shape (B, C, D, H, W). NOT sigmoidized.
            targets: Binary ground truth of shape (B, C, D, H, W).

        Returns:
            Scalar loss value.
        """
        assert logits.shape == targets.shape, (
            f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
        )

        # Apply label smoothing to targets for CE component
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets

        # Sigmoid activation for Dice computation
        pred_probs = torch.sigmoid(logits)
        num_channels = logits.shape[1]
        total_loss = torch.tensor(0.0, device=logits.device)

        for c in range(num_channels):
            pred_c = pred_probs[:, c]  # (B, D, H, W)
            target_c = targets[:, c]   # (B, D, H, W) — original for Dice
            smooth_target_c = smoothed_targets[:, c]  # Smoothed for CE
            logit_c = logits[:, c]     # (B, D, H, W) — raw for CE

            # Dice loss (using original hard targets)
            dice_loss = self._dice_loss(pred_c, target_c)

            # Binary CE loss (using smoothed targets)
            ce_loss = F.binary_cross_entropy_with_logits(
                logit_c, smooth_target_c, reduction='mean'
            )

            channel_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
            total_loss = total_loss + channel_loss

        return total_loss / num_channels

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss for a single channel.

        Args:
            pred: Predicted probabilities (B, D, H, W).
            target: Binary ground truth (B, D, H, W).

        Returns:
            Dice loss (1 - Dice coefficient).
        """
        # Flatten spatial dimensions
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice
