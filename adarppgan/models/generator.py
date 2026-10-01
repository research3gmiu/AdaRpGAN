"""
Resolution-flexible Generator for AdaRpGAN.

Architecture: ResNet-style with spectral norm + BatchNorm + self-attention.
Latent dim z=128 → 4×4 feature map → upsampled to img_size×img_size.

Supports: 32, 64, 128, 256 (any power-of-2 ≥ 8).
Self-attention is injected at the feature map closest to 32×32 for img_size ≥ 64.
Default behaviour (img_size=32) is identical to the original CIFAR-10 generator.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention2d


class ResBlockUp(nn.Module):
    """Residual block with bilinear upsampling (no checkerboard artifacts)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.skip  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(self.skip(x), scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.bn2(self.conv1(h)))
        h = self.conv2(h)
        return h + skip


class Generator(nn.Module):
    """
    Resolution-flexible Generator with optional self-attention.
    z (B, z_dim) → (B, 3, img_size, img_size) in [-1, 1].

    Parameters
    ----------
    z_dim       : int   Latent vector dimension (default 128).
    base_ch     : int   Base channel count (default 256).
    img_size    : int   Output spatial resolution; must be a power of 2 ≥ 8 (default 32).
    use_attn    : bool  Whether to add self-attention at ~32×32 resolution (default True for ≥64).
    """

    def __init__(self, z_dim: int = 128, base_ch: int = 256, img_size: int = 32,
                 use_attn: bool | None = None):
        super().__init__()
        assert img_size >= 8 and (img_size & (img_size - 1)) == 0, \
            f"img_size must be a power of 2 ≥ 8, got {img_size}"

        self.z_dim   = z_dim
        self.base_ch = base_ch
        self.img_size = img_size

        # Auto-enable attention for resolutions ≥ 64
        if use_attn is None:
            use_attn = (img_size >= 64)
        self.use_attn = use_attn

        # Number of 2× upsampling stages needed: 4 → img_size
        n_up = int(math.log2(img_size)) - 2   # 32→3, 64→4, 128→5, 256→6

        # Attention resolution: inject at the block that outputs ~32×32
        # Block i outputs spatial resolution 4 * 2^(i+1) = 8, 16, 32, 64, ...
        # So for 32×32 features, we want block index where 4*2^(i+1) = 32 → i = 2
        attn_block_idx = max(0, int(math.log2(32)) - 3)  # = 2 (the block outputting 32×32)

        # Project z to 4×4 feature map
        self.proj = nn.utils.spectral_norm(
            nn.Linear(z_dim, base_ch * 4 * 4, bias=False)
        )

        # Build upsampling blocks: each halves channels and doubles spatial dim
        layers = []
        ch_in = base_ch
        for i in range(n_up):
            ch_out = max(ch_in // 2, 32)  # floor at 32 channels
            layers.append(ResBlockUp(ch_in, ch_out))
            # Inject attention after the block that produces ~32×32 features
            if use_attn and i == attn_block_idx and n_up > 3:
                layers.append(SelfAttention2d(ch_out))
            ch_in = ch_out
        self.blocks = nn.Sequential(*layers)

        self.out = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ch_in, 3, 3, 1, 1, bias=True)),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.proj(z).view(B, self.base_ch, 4, 4)
        h = self.blocks(h)
        return self.out(h)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=device)
        return self(z)
