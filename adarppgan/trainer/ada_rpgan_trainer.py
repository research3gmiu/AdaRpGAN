"""
AdaRpGANTrainer
===============
Full training loop for the AdaRpGAN framework.

Features
--------
* EMA generator for smooth evaluation (decay=0.9999)
* DiffAugment pipeline (color + translation + cutout)
* Periodic FID evaluation during training
* Adaptive γ + n_critic via AdaptiveController
* Mixed-precision training (PyTorch 2.x API)
* Full checkpoint/resume support (models + optimisers + controller + EMA)

Usage
-----
    from trainer import AdaRpGANTrainer, TrainerConfig
    trainer = AdaRpGANTrainer(G, D, train_loader, cfg, device)
    trainer.fit()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from ..models import Generator, Discriminator, compute_gradient_penalty, compute_r1_penalty
from ..models.ema import ExponentialMovingAverage
from .losses import d_loss_rpgan, g_loss_rpgan
from .adaptive_controller import AdaptiveController, ControllerConfig

# PyTorch 2.x AMP API
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False


@dataclass
class TrainerConfig:
    # Optimiser
    lr_g:      float = 2e-4
    lr_d:      float = 4e-4      # TTUR: D lr higher than G
    beta1:     float = 0.0
    beta2:     float = 0.9

    # Batch & iterations
    batch_size: int  = 64
    n_epochs:   int  = 200

    # Gradient penalty variant: 'wgan-gp' | 'r1'
    gp_type:    str  = 'wgan-gp'

    # Adaptive controller config (passed to AdaptiveController)
    ctrl: ControllerConfig = field(default_factory=ControllerConfig)

    # EMA
    ema_decay: float = 0.9999

    # DiffAugment
    augment: str = "color,translation,cutout"   # "" to disable

    # Checkpoint / logging
    ckpt_dir:       str = "checkpoints"
    log_every:      int = 100     # log every N D-steps
    save_every:     int = 5       # save checkpoint every N epochs
    sample_every:   int = 5       # save sample grid every N epochs
    n_eval_samples: int = 50000   # FID sample count

    # Periodic FID evaluation
    eval_every: int = 0           # compute FID every N epochs (0 = disabled)
    eval_n_samples: int = 10000   # FID samples during training (smaller for speed)

    # Mixed precision
    amp: bool = True

    # Reproducibility
    seed: int = 42


class AdaRpGANTrainer:
    """
    Main AdaRpGAN trainer with EMA, DiffAugment, and periodic FID.
    """

    def __init__(
        self,
        G:           Generator,
        D:           Discriminator,
        loader:      torch.utils.data.DataLoader,
        cfg:         TrainerConfig,
        device:      torch.device,
        val_loader:  Optional[torch.utils.data.DataLoader] = None,
    ):
        self.G       = G.to(device)
        self.D       = D.to(device)
        self.loader  = loader
        self.cfg     = cfg
        self.device  = device
        self.val_loader = val_loader

        self.opt_G = Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_D = Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

        self.ctrl = AdaptiveController(cfg.ctrl)

        # EMA generator
        self.ema = ExponentialMovingAverage(self.G, decay=cfg.ema_decay)

        # DiffAugment
        self.augment = None
        if cfg.augment:
            from ..utils.diffaugment import DiffAugment
            self.augment = DiffAugment(policy=cfg.augment)

        # AMP — only functional on CUDA; no-op on MPS/CPU
        self._amp_enabled = cfg.amp and device.type == "cuda"
        if _AMP_NEW:
            self.scaler = GradScaler("cuda", enabled=self._amp_enabled)
        else:
            self.scaler = GradScaler(enabled=self._amp_enabled)

        self.global_d_step = 0
        self.epoch         = 0

        # Running metrics
        self.log: dict[str, list] = {
            "d_loss": [], "g_loss": [], "gamma": [],
            "n_critic": [], "grad_norm": [], "fid": [],
        }

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # Core step
    # ──────────────────────────────────────────

    def _autocast_ctx(self):
        """Return the correct autocast context for the device."""
        if _AMP_NEW:
            return autocast("cuda", enabled=self._amp_enabled)
        else:
            return autocast(enabled=self._amp_enabled)

    def train_step(self, real: torch.Tensor) -> dict:
        """One full GAN update: n_critic D steps + 1 G step."""
        cfg    = self.cfg
        G, D   = self.G, self.D
        device = self.device
        B      = real.size(0)
        real   = real.to(device)

        # ── D steps ─────────────────────────────
        d_losses   = []
        grad_norms = []

        for _ in range(self.ctrl.n_critic):
            self.opt_D.zero_grad(set_to_none=True)

            z = torch.randn(B, G.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            # Apply DiffAugment
            real_aug = self.augment(real) if self.augment else real
            fake_aug = self.augment(fake) if self.augment else fake

            with self._autocast_ctx():
                d_real = D(real_aug)
                d_fake = D(fake_aug)
                adv    = d_loss_rpgan(d_real, d_fake)

            # Gradient penalty (outside autocast for numerical stability)
            if cfg.gp_type == 'wgan-gp':
                gp, gn = compute_gradient_penalty(D, real, fake, device)
            else:
                gp, gn = compute_r1_penalty(D, real)

            gamma  = self.ctrl.gamma
            d_loss = adv + gamma * gp

            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()

            d_losses.append(d_loss.item())
            grad_norms.append(gn.item())
            self.global_d_step += 1

        # Observe: update γ and n_critic
        mean_gn    = sum(grad_norms) / len(grad_norms)
        mean_dloss = sum(d_losses) / len(d_losses)
        self.ctrl.observe(grad_norm=mean_gn, d_loss=mean_dloss)

        # ── G step ──────────────────────────────
        self.opt_G.zero_grad(set_to_none=True)

        z = torch.randn(B, G.z_dim, device=device)
        with self._autocast_ctx():
            fake   = G(z)
            fake_aug = self.augment(fake) if self.augment else fake
            d_real = D(real_aug if self.augment else real).detach()
            d_fake = D(fake_aug)
            g_loss = g_loss_rpgan(d_real, d_fake)

        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_G)
        self.scaler.update()

        # Update EMA
        self.ema.update()

        return {
            "d_loss":    mean_dloss,
            "g_loss":    g_loss.item(),
            "grad_norm": mean_gn,
            "gamma":     self.ctrl.gamma,
            "n_critic":  self.ctrl.n_critic,
        }

    # ──────────────────────────────────────────
    # Epoch & full training loop
    # ──────────────────────────────────────────

    def train_epoch(self) -> dict:
        self.G.train(); self.D.train()
        metrics_sum = {"d_loss": 0, "g_loss": 0, "grad_norm": 0,
                       "gamma": 0, "n_critic": 0}
        n_steps = 0

        for batch in self.loader:
            real = batch[0] if isinstance(batch, (list, tuple)) else batch
            m    = self.train_step(real)
            for k in metrics_sum:
                metrics_sum[k] += m[k]
            n_steps += 1

            if self.global_d_step % self.cfg.log_every == 0:
                ctrl_info = self.ctrl.info()
                print(
                    f"  [D-step {self.global_d_step:6d}]  "
                    f"d={m['d_loss']:.4f}  g={m['g_loss']:.4f}  "
                    f"γ={ctrl_info['gamma']:.3f}  "
                    f"n_c={ctrl_info['n_critic']}  "
                    f"|∇|={ctrl_info['viol_ema']:.4f}  "
                    f"var={ctrl_info['loss_var']:.4f}"
                )

        for k in metrics_sum:
            metrics_sum[k] /= max(n_steps, 1)
        return metrics_sum

    def fit(self, n_epochs: Optional[int] = None) -> None:
        n_epochs = n_epochs or self.cfg.n_epochs
        print(f"[AdaRpGAN] Starting training: {n_epochs} epochs, device={self.device}")
        print(f"  γ₀={self.ctrl.gamma:.1f}  n_critic₀={self.ctrl.n_critic}")
        print(f"  EMA decay={self.cfg.ema_decay}")
        if self.augment:
            print(f"  DiffAugment: {self.cfg.augment}")

        for ep in range(1, n_epochs + 1):
            self.epoch = ep
            t0 = time.time()
            m  = self.train_epoch()
            dt = time.time() - t0

            print(
                f"Epoch {ep:4d}/{n_epochs}  "
                f"d={m['d_loss']:.4f}  g={m['g_loss']:.4f}  "
                f"γ={m['gamma']:.3f}  n_c={m['n_critic']:.1f}  "
                f"[{dt:.1f}s]"
            )

            for k in ["d_loss", "g_loss", "gamma", "n_critic", "grad_norm"]:
                self.log[k].append(m.get(k, 0))

            if ep % self.cfg.save_every == 0:
                self.save_checkpoint(f"epoch_{ep:04d}.pt")

            if ep % self.cfg.sample_every == 0:
                self._save_samples(ep)

            # Periodic FID evaluation
            if self.cfg.eval_every > 0 and ep % self.cfg.eval_every == 0:
                fid = self._compute_fid()
                self.log["fid"].append(fid)
                print(f"  [FID] epoch {ep}: {fid:.2f}")

        self.save_checkpoint("final.pt")
        print("[AdaRpGAN] Training complete.")

    # ──────────────────────────────────────────
    # FID evaluation (uses EMA generator)
    # ──────────────────────────────────────────

    def _compute_fid(self) -> float:
        """Compute FID using EMA generator weights."""
        from ..utils.metrics import compute_fid
        loader = self.val_loader or self.loader
        self.ema.apply()
        try:
            fid = compute_fid(
                self.G, loader,
                n_samples=self.cfg.eval_n_samples,
                batch_size=min(256, self.cfg.batch_size * 4),
                device=self.device,
            )
        finally:
            self.ema.restore()
        return fid

    # ──────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────

    def save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.cfg.ckpt_dir, name)
        torch.save({
            "epoch":           self.epoch,
            "global_d_step":   self.global_d_step,
            "G_state":         self.G.state_dict(),
            "D_state":         self.D.state_dict(),
            "opt_G":           self.opt_G.state_dict(),
            "opt_D":           self.opt_D.state_dict(),
            "ctrl":            self.ctrl.state_dict(),
            "ema":             self.ema.state_dict(),
            "log":             self.log,
        }, path)
        print(f"  ✓ checkpoint → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.epoch         = ckpt["epoch"]
        self.global_d_step = ckpt["global_d_step"]
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.ctrl.load_state_dict(ckpt["ctrl"])
        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        self.log = ckpt["log"]
        print(f"  ✓ resumed from {path} (epoch {self.epoch})")

    # ──────────────────────────────────────────
    # Sample saving (uses EMA generator)
    # ──────────────────────────────────────────

    def _save_samples(self, epoch: int, n: int = 64) -> None:
        import torchvision.utils as vutils
        self.ema.apply()
        self.G.eval()
        with torch.no_grad():
            imgs = self.G.sample(n, self.device).cpu()
        imgs = (imgs + 1.0) / 2.0  # [-1,1] → [0,1]
        path = os.path.join(self.cfg.ckpt_dir, f"samples_epoch_{epoch:04d}.png")
        vutils.save_image(imgs, path, nrow=8, padding=2)
        self.G.train()
        self.ema.restore()
        print(f"  ✓ samples    → {path}")
