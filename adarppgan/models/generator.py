"""
Resolution-flexible Generator for AdaRpGAN.

Architecture: ResNet-style with spectral norm + conditional BatchNorm + self-attention.
Latent dim z=128 → 4×4 feature map → upsampled to img_size×img_size.

Supports: 32, 64, 128, 256 (any power-of-2 ≥ 8).
Self-attention is injected at the feature map closest to 32×32 for img_size ≥ 64.

Changes from v1:
  - Conditional Batch Normalization (CBN) for class-conditioned generation
  - Orthogonal initialisation for all conv/linear layers
  - Wider default base channels (512)
  - Supports both conditional (n_classes > 0) and unconditional mode
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention2d


# ──────────────────────────────────────────────
# Conditional Batch Normalisation
# ──────────────────────────────────────────────

class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalisation (de Vries et al., 2017).

    Standard BN but γ and β are predicted from a class embedding
    rather than being free parameters.

    Parameters
    ----------
    num_features : int   Number of channels.
    n_classes    : int   Number of classes (for the embedding table).
    """

    def __init__(self, num_features: int, n_classes: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # Learnable per-class affine parameters
        self.gain = nn.Embedding(n_classes, num_features)
        self.bias = nn.Embedding(n_classes, num_features)

        # Initialise gain to 1 and bias to 0 (identity transform)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) feature maps
        y: (B,) class labels (long)
        """
        out = self.bn(x)
        gamma = self.gain(y).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.bias(y).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


# ──────────────────────────────────────────────
# Residual Upsampling Block
# ──────────────────────────────────────────────

class ResBlockUp(nn.Module):
    """Residual block with bilinear upsampling (no checkerboard artifacts).

    Supports both conditional and unconditional BN.
    """

    def __init__(self, in_ch: int, out_ch: int, n_classes: int = 0):
        super().__init__()
        self.conditional = n_classes > 0

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.skip  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))

        if self.conditional:
            self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        else:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        skip = F.interpolate(self.skip(x), scale_factor=2, mode="bilinear", align_corners=False)

        if self.conditional and y is not None:
            h = F.relu(self.bn1(x, y))
        else:
            h = F.relu(self.bn1(x))

        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)

        if self.conditional and y is not None:
            h = F.relu(self.bn2(self.conv1(h), y))
        else:
            h = F.relu(self.bn2(self.conv1(h)))

        h = self.conv2(h)
        return h + skip


# ──────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────

class Generator(nn.Module):
    """
    Resolution-flexible Generator with optional self-attention and
    class conditioning.

    z (B, z_dim) [+ y (B,)] → (B, 3, img_size, img_size) in [-1, 1].

    Parameters
    ----------
    z_dim       : int   Latent vector dimension (default 128).
    base_ch     : int   Base channel count (default 512).
    img_size    : int   Output spatial resolution; must be a power of 2 ≥ 8 (default 32).
    n_classes   : int   Number of classes for conditioning (0 = unconditional).
    use_attn    : bool  Whether to add self-attention at ~32×32 resolution.
    """

    def __init__(self, z_dim: int = 128, base_ch: int = 512, img_size: int = 32,
                 n_classes: int = 0, use_attn: bool | None = None):
        super().__init__()
        assert img_size >= 8 and (img_size & (img_size - 1)) == 0, \
            f"img_size must be a power of 2 ≥ 8, got {img_size}"

        self.z_dim     = z_dim
        self.base_ch   = base_ch
        self.img_size  = img_size
        self.n_classes = n_classes
        self.conditional = n_classes > 0

        # Auto-enable attention for resolutions ≥ 64
        if use_attn is None:
            use_attn = (img_size >= 64)
        self.use_attn = use_attn

        # Number of 2× upsampling stages needed: 4 → img_size
        n_up = int(math.log2(img_size)) - 2   # 32→3, 64→4, 128→5, 256→6

        # Attention resolution: inject at the block that outputs ~32×32
        attn_block_idx = max(0, int(math.log2(32)) - 3)  # = 2

        # Project z to 4×4 feature map
        self.proj = nn.utils.spectral_norm(
            nn.Linear(z_dim, base_ch * 4 * 4, bias=False)
        )

        # Build upsampling blocks: each halves channels and doubles spatial dim
        self.blocks = nn.ModuleList()
        self.attn_indices = set()
        ch_in = base_ch
        for i in range(n_up):
            ch_out = max(ch_in // 2, 64)  # floor at 64 channels
            self.blocks.append(ResBlockUp(ch_in, ch_out, n_classes=n_classes))
            # Inject attention after the block that produces ~32×32 features
            if use_attn and i == attn_block_idx and n_up > 3:
                self.attn_indices.add(i)
                self.blocks.append(SelfAttention2d(ch_out))
            ch_in = ch_out

        self.final_ch = ch_in

        if self.conditional:
            self.out_bn = ConditionalBatchNorm2d(ch_in, n_classes)
        else:
            self.out_bn = nn.BatchNorm2d(ch_in)

        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ch_in, 3, 3, 1, 1, bias=True)),
            nn.Tanh(),
        )

        # Orthogonal initialisation
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                # Already initialised in CBN
                pass

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B = z.size(0)
        h = self.proj(z).view(B, self.base_ch, 4, 4)

        for block in self.blocks:
            if isinstance(block, ResBlockUp):
                h = block(h, y)
            else:
                h = block(h)  # SelfAttention2d

        if self.conditional and y is not None:
            h = self.out_bn(h, y)
        else:
            h = self.out_bn(h)

        return self.out_conv(h)

    def sample(self, n: int, device: torch.device,
               y: torch.Tensor | None = None) -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=device)
        if self.conditional and y is None:
            y = torch.randint(0, self.n_classes, (n,), device=device)
        return self(z, y)
