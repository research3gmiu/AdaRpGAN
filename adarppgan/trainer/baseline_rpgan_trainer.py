"""
BaselineRpGANTrainer (v2)
=========================
Fixed-γ RpGAN/Hinge trainer — the comparison baseline.
Now includes EMA generator, class conditioning, cosine LR,
and lazy R1 for fair comparison with AdaRpGAN v2.
"""

from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ..models import Generator, Discriminator, compute_gradient_penalty, compute_r1_penalty
from ..models.ema import ExponentialMovingAverage
from .losses import get_loss_fn

try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False


def cosine_warmup_schedule(warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@dataclass
class BaselineConfig:
    lr_g:      float = 2e-4
    lr_d:      float = 2e-4
    beta1:     float = 0.0
    beta2:     float = 0.999
    eps:       float = 1e-4
    batch_size: int  = 128
    n_epochs:   int  = 800

    gamma:    float = 10.0   # Fixed GP weight
    n_critic: int   = 1      # Fixed D steps per G step

    loss_type: str  = 'hinge'
    gp_type:   str  = 'r1'
    r1_gamma:  float = 10.0
    r1_interval: int = 16

    n_classes: int = 10

    ema_decay: float = 0.9999
    augment:   str   = "color,translation,cutout"

    warmup_epochs: int = 5
    use_cosine_lr: bool = True

    ckpt_dir:    str = "checkpoints_baseline"
    log_every:   int = 100
    save_every:  int = 50
    sample_every: int = 25
    amp: bool = True


class BaselineRpGANTrainer:

    def __init__(self, G, D, loader, cfg: BaselineConfig, device: torch.device):
        self.G, self.D   = G.to(device), D.to(device)
        self.loader      = loader
        self.cfg         = cfg
        self.device      = device
        self.conditional = cfg.n_classes > 0

        self.opt_G = Adam(G.parameters(), lr=cfg.lr_g,
                          betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.opt_D = Adam(D.parameters(), lr=cfg.lr_d,
                          betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)

        self.sched_G = None
        self.sched_D = None
        if cfg.use_cosine_lr:
            lr_fn = cosine_warmup_schedule(cfg.warmup_epochs, cfg.n_epochs)
            self.sched_G = LambdaLR(self.opt_G, lr_fn)
            self.sched_D = LambdaLR(self.opt_D, lr_fn)

        self.d_loss_fn, self.g_loss_fn = get_loss_fn(cfg.loss_type)

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

    def train_step(self, real: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        cfg    = self.cfg
        G, D   = self.G, self.D
        device = self.device
        B      = real.size(0)
        real   = real.to(device)

        y = None
        if self.conditional and labels is not None:
            y = labels.to(device)

        real_aug = self.augment(real) if self.augment else real

        # D steps
        d_losses = []
        for _ in range(cfg.n_critic):
            self.opt_D.zero_grad(set_to_none=True)
            z = torch.randn(B, G.z_dim, device=device)

            y_fake = None
            if self.conditional:
                y_fake = torch.randint(0, cfg.n_classes, (B,), device=device)

            with torch.no_grad():
                fake = G(z, y_fake)

            fake_aug = self.augment(fake) if self.augment else fake

            with self._autocast_ctx():
                d_real = D(real_aug, y)
                d_fake = D(fake_aug, y_fake)
                adv    = self.d_loss_fn(d_real, d_fake)

            gp = torch.tensor(0.0, device=device)
            if cfg.gp_type == 'r1':
                if self.global_d_step % cfg.r1_interval == 0:
                    gp, _ = compute_r1_penalty(D, real, y)
                    gp = gp * cfg.r1_gamma * cfg.r1_interval
            else:
                gp, _ = compute_gradient_penalty(D, real, fake, device, y)

            gamma_w = cfg.gamma if cfg.gp_type != 'r1' else 1.0
            d_loss = adv + gamma_w * gp
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()
            d_losses.append(d_loss.item())
            self.global_d_step += 1

        # G step
        self.opt_G.zero_grad(set_to_none=True)
        z = torch.randn(B, G.z_dim, device=device)

        y_fake_g = None
        if self.conditional:
            y_fake_g = torch.randint(0, cfg.n_classes, (B,), device=device)

        with self._autocast_ctx():
            fake     = G(z, y_fake_g)
            fake_aug = self.augment(fake) if self.augment else fake
            d_real_g = D(real_aug, y).detach()
            d_fake_g = D(fake_aug, y_fake_g)
            g_loss   = self.g_loss_fn(d_real_g, d_fake_g)
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
            if isinstance(batch, (list, tuple)):
                real = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                real = batch
                labels = None
            m    = self.train_step(real, labels)
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
        print(f"[Baseline v2] γ={self.cfg.gamma} (fixed), "
              f"n_critic={self.cfg.n_critic} (fixed), "
              f"loss={self.cfg.loss_type}")
        for ep in range(1, n_epochs + 1):
            self.epoch = ep
            t0 = time.time()
            m  = self.train_epoch()
            dt = time.time() - t0

            if self.sched_G is not None:
                self.sched_G.step()
                self.sched_D.step()

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
        state = {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "ema": self.ema.state_dict(),
            "log": self.log,
        }
        if self.sched_G is not None:
            state["sched_G"] = self.sched_G.state_dict()
            state["sched_D"] = self.sched_D.state_dict()
        torch.save(state, path)
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
