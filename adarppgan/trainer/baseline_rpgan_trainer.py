"""
BaselineRpGANTrainer
====================
Fixed-γ RpGAN trainer — the comparison baseline.
Now includes EMA generator for fair comparison with AdaRpGAN.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch.optim import Adam

from ..models import Generator, Discriminator, compute_gradient_penalty, compute_r1_penalty
from ..models.ema import ExponentialMovingAverage
from .losses import d_loss_rpgan, g_loss_rpgan

try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False


@dataclass
class BaselineConfig:
    lr_g:      float = 2e-4
    lr_d:      float = 4e-4
    beta1:     float = 0.0
    beta2:     float = 0.9
    batch_size: int  = 64
    n_epochs:   int  = 200

    gamma:    float = 10.0   # Fixed GP weight
    n_critic: int   = 1      # Fixed D steps per G step
    gp_type:  str   = 'wgan-gp'

    ema_decay: float = 0.9999
    augment:   str   = "color,translation,cutout"

    ckpt_dir:    str = "checkpoints_baseline"
    log_every:   int = 100
    save_every:  int = 5
    sample_every: int = 5
    amp: bool = True


class BaselineRpGANTrainer:

    def __init__(self, G, D, loader, cfg: BaselineConfig, device: torch.device):
        self.G, self.D   = G.to(device), D.to(device)
        self.loader      = loader
        self.cfg         = cfg
        self.device      = device

        self.opt_G = Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_D = Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

        self.ema = ExponentialMovingAverage(self.G, decay=cfg.ema_decay)

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
        self.log: dict     = {"d_loss": [], "g_loss": [], "fid": []}

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def _autocast_ctx(self):
        if _AMP_NEW:
            return autocast("cuda", enabled=self._amp_enabled)
        else:
            return autocast(enabled=self._amp_enabled)

    def train_step(self, real: torch.Tensor) -> dict:
        cfg    = self.cfg
        G, D   = self.G, self.D
        device = self.device
        B      = real.size(0)
        real   = real.to(device)

        # Always define real_aug before the D loop so the G step can safely use it.
        real_aug = self.augment(real) if self.augment else real

        # D steps
        d_losses = []
        for _ in range(cfg.n_critic):
            self.opt_D.zero_grad(set_to_none=True)
            z = torch.randn(B, G.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            fake_aug = self.augment(fake) if self.augment else fake

            with self._autocast_ctx():
                d_real = D(real_aug)
                d_fake = D(fake_aug)
                adv    = d_loss_rpgan(d_real, d_fake)

            if cfg.gp_type == 'wgan-gp':
                gp, _ = compute_gradient_penalty(D, real, fake, device)
            else:
                gp, _ = compute_r1_penalty(D, real)

            d_loss = adv + cfg.gamma * gp
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()
            d_losses.append(d_loss.item())
            self.global_d_step += 1

        # G step
        self.opt_G.zero_grad(set_to_none=True)
        z = torch.randn(B, G.z_dim, device=device)
        with self._autocast_ctx():
            fake     = G(z)
            fake_aug = self.augment(fake) if self.augment else fake
            d_real   = D(real_aug).detach()
            d_fake   = D(fake_aug)
            g_loss   = g_loss_rpgan(d_real, d_fake)
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_G)
        self.scaler.update()

        self.ema.update()

        return {
            "d_loss": sum(d_losses) / len(d_losses),
            "g_loss": g_loss.item(),
        }

    def train_epoch(self) -> dict:
        self.G.train(); self.D.train()
        sums  = {"d_loss": 0.0, "g_loss": 0.0}
        steps = 0
        for batch in self.loader:
            real = batch[0] if isinstance(batch, (list, tuple)) else batch
            m    = self.train_step(real)
            sums["d_loss"] += m["d_loss"]
            sums["g_loss"] += m["g_loss"]
            steps += 1
            if self.global_d_step % self.cfg.log_every == 0:
                print(f"  [baseline D-step {self.global_d_step:6d}]  "
                      f"d={m['d_loss']:.4f}  g={m['g_loss']:.4f}  "
                      f"γ={self.cfg.gamma:.1f} (fixed)  "
                      f"n_c={self.cfg.n_critic} (fixed)")
        return {k: v / max(steps, 1) for k, v in sums.items()}

    def fit(self, n_epochs=None):
        n_epochs = n_epochs or self.cfg.n_epochs
        print(f"[Baseline RpGAN] γ={self.cfg.gamma} (fixed), "
              f"n_critic={self.cfg.n_critic} (fixed)")
        for ep in range(1, n_epochs + 1):
            self.epoch = ep
            t0 = time.time()
            m  = self.train_epoch()
            dt = time.time() - t0
            print(f"Epoch {ep:4d}/{n_epochs}  d={m['d_loss']:.4f}  "
                  f"g={m['g_loss']:.4f}  [{dt:.1f}s]")
            for k in ["d_loss", "g_loss"]:
                self.log[k].append(m[k])
            if ep % self.cfg.save_every == 0:
                self._save(f"epoch_{ep:04d}.pt")
            if ep % self.cfg.sample_every == 0:
                self._save_samples(ep)
        self._save("final.pt")

    def _save(self, name):
        path = os.path.join(self.cfg.ckpt_dir, name)
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "ema": self.ema.state_dict(),
            "log": self.log,
        }, path)
        print(f"  ✓ baseline checkpoint → {path}")

    def _save_samples(self, epoch: int, n: int = 64) -> None:
        import torchvision.utils as vutils
        self.ema.apply()
        self.G.eval()
        with torch.no_grad():
            imgs = self.G.sample(n, self.device).cpu()
        imgs = (imgs + 1.0) / 2.0
        path = os.path.join(self.cfg.ckpt_dir, f"samples_epoch_{epoch:04d}.png")
        vutils.save_image(imgs, path, nrow=8, padding=2)
        self.G.train()
        self.ema.restore()
        print(f"  ✓ samples → {path}")
