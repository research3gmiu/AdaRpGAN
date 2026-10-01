"""
Relativistic GAN (RpGAN) loss functions.

Reference: Jolicoeur-Martineau (2019) "The relativistic discriminator:
a key element missing from standard GAN."

Standard (non-relativistic) GAN: D tries to distinguish real from fake.
Relativistic GAN:                D predicts the *relative* realness —
                                 P(real is more realistic than fake).

RpGAN losses used here
----------------------
D loss:  E[-log σ(D(real) - E[D(fake)])]
       + E[-log σ(E[D(real)] - D(fake))]

G loss:  E[-log σ(D(fake)  - E[D(real)])]
       + E[-log σ(E[D(fake)] - D(real))]

With gradient penalty appended to D loss.
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
# Vanilla GAN losses (for ablation / baseline)
# ──────────────────────────────────────────────

def d_loss_vanilla(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return (loss_real + loss_fake) * 0.5


def g_loss_vanilla(d_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
