"""
Differentiable Augmentation for GANs (DiffAugment).

Reference: Zhao et al. (2020) "Differentiable Augmentation for Data-Efficient GAN Training"

Applied to BOTH real and fake images before D sees them.
All operations are differentiable, so gradients flow through augmentation.
This prevents D from overfitting, especially on small datasets.

Usage
-----
    from utils.diffaugment import DiffAugment
    aug = DiffAugment(policy="color,translation,cutout")
    real_aug = aug(real_images)
    fake_aug = aug(fake_images)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffAugment:
    """
    Differentiable augmentation pipeline.

    Parameters
    ----------
    policy : str
        Comma-separated list of augmentation types.
        Options: "color", "translation", "cutout"
        Default: "color,translation,cutout" (all three)
    """

    def __init__(self, policy: str = "color,translation,cutout"):
        self.policies = [p.strip() for p in policy.split(",") if p.strip()]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for p in self.policies:
            if p == "color":
                x = _augment_color(x)
            elif p == "translation":
                x = _augment_translation(x)
            elif p == "cutout":
                x = _augment_cutout(x)
        return x


def _augment_color(x: torch.Tensor) -> torch.Tensor:
    """Random brightness, saturation, and contrast."""
    B, C, H, W = x.shape

    # Brightness
    x = x + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)

    # Saturation
    mean = x.mean(dim=1, keepdim=True)
    x = mean + (x - mean) * (torch.rand(B, 1, 1, 1, device=x.device) * 2.0)

    # Contrast
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = mean + (x - mean) * (torch.rand(B, 1, 1, 1, device=x.device) + 0.5)

    return x


def _augment_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random translation with reflection padding."""
    B, C, H, W = x.shape
    shift = int(H * ratio + 0.5)

    tx = torch.randint(-shift, shift + 1, (B, 1, 1), device=x.device).float()
    ty = torch.randint(-shift, shift + 1, (B, 1, 1), device=x.device).float()

    # Build affine grid for translation
    grid_x = torch.arange(W, device=x.device).float()
    grid_y = torch.arange(H, device=x.device).float()
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    grid_x = (grid_x.unsqueeze(0) + tx) / (W - 1) * 2 - 1  # (B, H, W)
    grid_y = (grid_y.unsqueeze(0) + ty) / (H - 1) * 2 - 1

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=True)


def _augment_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Random rectangular cutout."""
    B, C, H, W = x.shape
    cut_h = int(H * ratio + 0.5)
    cut_w = int(W * ratio + 0.5)

    offset_y = torch.randint(0, H - cut_h + 1, (B,), device=x.device)
    offset_x = torch.randint(0, W - cut_w + 1, (B,), device=x.device)

    grid_y = torch.arange(H, device=x.device).unsqueeze(0)  # (1, H)
    grid_x = torch.arange(W, device=x.device).unsqueeze(0)

    mask_y = ((grid_y >= offset_y.unsqueeze(1)) &
              (grid_y < (offset_y + cut_h).unsqueeze(1)))  # (B, H)
    mask_x = ((grid_x >= offset_x.unsqueeze(1)) &
              (grid_x < (offset_x + cut_w).unsqueeze(1)))  # (B, W)

    mask = mask_y.unsqueeze(2) & mask_x.unsqueeze(1)  # (B, H, W)
    mask = mask.unsqueeze(1).float()  # (B, 1, H, W)

    return x * (1.0 - mask)
