"""
Self-Attention layer for GAN architectures.

Implements spectral-normalised self-attention (SAGAN, Zhang et al. 2019).
Injected at intermediate feature map resolutions (typically 16×16 or 32×32)
to capture long-range spatial dependencies that convolutions miss.

Usage
-----
    from models.attention import SelfAttention2d
    attn = SelfAttention2d(ch=256)
    out = attn(feature_map)  # (B, 256, 16, 16) → (B, 256, 16, 16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    """
    Spectral-normalised self-attention for 2D feature maps.

    Parameters
    ----------
    ch : int
        Number of input channels.
    reduction : int
        Channel reduction factor for Q and K (saves memory). Default 8.
    """

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.ch = ch
        mid = max(ch // reduction, 1)

        self.query = nn.utils.spectral_norm(nn.Conv2d(ch, mid, 1, bias=False))
        self.key   = nn.utils.spectral_norm(nn.Conv2d(ch, mid, 1, bias=False))
        self.value = nn.utils.spectral_norm(nn.Conv2d(ch, ch,  1, bias=False))

        # Learnable residual weight — starts at 0 so attention is initially a no-op
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N)           # (B, mid, N)
        k = self.key(x).view(B, -1, N)              # (B, mid, N)
        v = self.value(x).view(B, C, N)              # (B, C, N)

        # Attention weights
        attn = torch.bmm(q.transpose(1, 2), k)      # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))    # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x
