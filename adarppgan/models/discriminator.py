"""
Resolution-flexible Discriminator for AdaRpGAN.

Architecture: Residual SN-GAN discriminator with optional self-attention.
Returns raw logits (no sigmoid) for relativistic loss computation.

Supports: 32, 64, 128, 256 (any power-of-2 ≥ 8).
Self-attention is injected at the ~16×16 feature map for img_size ≥ 64.
Default behaviour (img_size=32) is identical to the original CIFAR-10 discriminator.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .attention import SelfAttention2d


class ResBlockDown(nn.Module):
    """Residual block with average-pool downsampling."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.skip  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2) if self.downsample else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self._pool(self.skip(x))
        h = F.leaky_relu(x, 0.2, inplace=False)
        h = F.leaky_relu(self.conv1(h), 0.2, inplace=True)
        h = self._pool(self.conv2(h))
        return h + skip


class Discriminator(nn.Module):
    """
    Resolution-flexible Discriminator with optional self-attention.
    (B, 3, img_size, img_size) → (B, 1) raw logits.

    Parameters
    ----------
    base_ch   : int   Base channel count (default 128).
    img_size  : int   Input spatial resolution; must be a power of 2 ≥ 8 (default 32).
    use_attn  : bool  Whether to add self-attention (default True for ≥64).
    """

    def __init__(self, base_ch: int = 128, img_size: int = 32,
                 use_attn: bool | None = None):
        super().__init__()
        assert img_size >= 8 and (img_size & (img_size - 1)) == 0, \
            f"img_size must be a power of 2 ≥ 8, got {img_size}"

        self.base_ch  = base_ch
        self.img_size = img_size

        if use_attn is None:
            use_attn = (img_size >= 64)
        self.use_attn = use_attn

        # Number of downsampling stages: img_size → 4
        n_down = int(math.log2(img_size)) - 2   # 32→3, 64→4, 128→5, 256→6

        # Attention: inject after the block that produces ~16×16 features
        # Block i takes spatial from img_size/2^i → img_size/2^(i+1)
        # For 64: block 0 → 32, block 1 → 16 ← attention here
        # For 128: block 0 → 64, block 1 → 32, block 2 → 16 ← attention here
        attn_after_block = -1
        if use_attn:
            target_spatial = 16
            for i in range(n_down):
                output_spatial = img_size // (2 ** (i + 1))
                if output_spatial == target_spatial:
                    attn_after_block = i
                    break
            if attn_after_block < 0 and n_down > 3:
                attn_after_block = 1  # fallback

        # Build blocks
        layers = []
        ch_in = 3
        for i in range(n_down):
            ch_out = base_ch * min(2 ** i, 8)  # cap at 8× base
            layers.append(ResBlockDown(ch_in, ch_out, downsample=True))
            if i == attn_after_block:
                layers.append(SelfAttention2d(ch_out))
            ch_in = ch_out

        # Final non-downsampling block at 4×4
        ch_out = base_ch * min(2 ** n_down, 8)
        layers.append(ResBlockDown(ch_in, ch_out, downsample=False))
        self.blocks = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.utils.spectral_norm(nn.Linear(ch_out, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self.pool(h).view(h.size(0), -1)
        return self.head(h)


# ──────────────────────────────────────────────
# Gradient Penalty Utilities
# ──────────────────────────────────────────────

def compute_gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Two-sided gradient penalty (WGAN-GP style).

    Returns
    -------
    gp        : scalar gradient-penalty loss
    grad_norm : mean ||∇D(x̂)||₂ over the batch (for adaptive controller)
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

    d_hat = D(x_hat)
    grads = autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads_flat  = grads.view(B, -1)
    grad_norm   = grads_flat.norm(2, dim=1)           # (B,)
    gp          = ((grad_norm - 1.0) ** 2).mean()
    return gp, grad_norm.mean().detach()


def compute_r1_penalty(
    D: nn.Module,
    real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    R1 regularisation (one-sided GP on real data only).
    """
    real_req = real.requires_grad_(True)
    d_real   = D(real_req)
    grads    = autograd.grad(
        outputs=d_real.sum(),
        inputs=real_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads_flat = grads.view(real.size(0), -1)
    grad_norm  = grads_flat.norm(2, dim=1)
    r1         = (grad_norm ** 2).mean() * 0.5
    return r1, grad_norm.mean().detach()
