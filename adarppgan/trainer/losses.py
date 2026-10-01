"""
GAN loss functions for AdaRpGAN.

Includes:
  - Relativistic GAN (RpGAN) losses — Jolicoeur-Martineau (2019)
  - Hinge losses — Lim & Ye (2017), Miyato et al. (2018)
  - Vanilla GAN losses (for ablation)

Hinge loss + spectral norm + R1 regularisation is the combination used by
most state-of-the-art GANs (BigGAN, StyleGAN2, R3GAN).
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Core RpGAN
# ──────────────────────────────────────────────

def d_loss_rpgan(
    d_real: torch.Tensor,   # (B,1) logits on real samples
    d_fake: torch.Tensor,   # (B,1) logits on fake samples
) -> torch.Tensor:
    """Relativistic average discriminator loss (scalar)."""
    real_mean = d_real.mean()
    fake_mean = d_fake.mean()

    loss_real = F.binary_cross_entropy_with_logits(
        d_real - fake_mean,
        torch.ones_like(d_real),
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake - real_mean,
        torch.zeros_like(d_fake),
    )
    return (loss_real + loss_fake) * 0.5


def g_loss_rpgan(
    d_real: torch.Tensor,   # (B,1) logits on real — NO grad needed
    d_fake: torch.Tensor,   # (B,1) logits on fake — grad flows through G
) -> torch.Tensor:
    """Relativistic average generator loss (scalar)."""
    real_mean = d_real.mean()
    fake_mean = d_fake.mean()

    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake - real_mean,
        torch.ones_like(d_fake),
    )
    loss_real = F.binary_cross_entropy_with_logits(
        d_real - fake_mean,
        torch.zeros_like(d_real),
    )
    return (loss_fake + loss_real) * 0.5


# ──────────────────────────────────────────────
# Hinge losses (SOTA standard)
# ──────────────────────────────────────────────

def d_loss_hinge(
    d_real: torch.Tensor,   # (B,1) logits on real samples
    d_fake: torch.Tensor,   # (B,1) logits on fake samples
) -> torch.Tensor:
    """
    Hinge loss for discriminator.

    D wants: D(real) > +1 and D(fake) < -1.
    L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
    """
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def g_loss_hinge(
    d_fake: torch.Tensor,   # (B,1) logits on fake — grad flows through G
) -> torch.Tensor:
    """
    Hinge loss for generator.

    G wants: D(fake) to be high (positive).
    L_G = -E[D(fake)]
    """
    return -d_fake.mean()


# ──────────────────────────────────────────────
# Vanilla GAN losses (for ablation / baseline)
# ──────────────────────────────────────────────

def d_loss_vanilla(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return (loss_real + loss_fake) * 0.5


def g_loss_vanilla(d_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


# ──────────────────────────────────────────────
# Loss selector
# ──────────────────────────────────────────────

def get_loss_fn(loss_type: str = "hinge"):
    """
    Returns (d_loss_fn, g_loss_fn) callables.

    Parameters
    ----------
    loss_type : str  One of 'hinge', 'rpgan', 'vanilla'.

    Returns
    -------
    d_loss_fn : callable(d_real, d_fake) -> scalar
    g_loss_fn : callable(d_real_or_none, d_fake) -> scalar
        For hinge/vanilla, g_loss_fn ignores the first argument.
        For rpgan, g_loss_fn uses both arguments.
    """
    if loss_type == "hinge":
        def _g(d_real, d_fake):
            return g_loss_hinge(d_fake)
        return d_loss_hinge, _g
    elif loss_type == "rpgan":
        return d_loss_rpgan, g_loss_rpgan
    elif loss_type == "vanilla":
        def _g(d_real, d_fake):
            return g_loss_vanilla(d_fake)
        return d_loss_vanilla, _g
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'hinge', 'rpgan', or 'vanilla'.")
