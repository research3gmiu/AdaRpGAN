#!/usr/bin/env python3
"""
quickstart.py — Minimal AdaRpGAN usage example.

Run from anywhere (project root or examples/ folder):

    # From project root:
    ./venv/bin/python examples/quickstart.py

    # Or after pip install -e .:
    python examples/quickstart.py

This script will:
  1. Load CIFAR-10 (auto-downloads on first run)
  2. Create Generator and Discriminator
  3. Train AdaRpGAN for 5 epochs (quick demo)
  4. Save sample images and training curves
"""

import os
import sys

# ── Make the package importable from any working directory ────────────
# Adds the project root (parent of examples/) to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # project root, contains adarppgan/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from adarppgan.models import Generator, Discriminator
from adarppgan.trainer import (
    AdaRpGANTrainer,
    TrainerConfig,
    ControllerConfig,
)
from adarppgan.utils import (
    get_cifar10_loaders,
    plot_training_curves,
    plot_controller_state,
)


def main():
    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = get_cifar10_loaders(batch_size=64)

    # ── Models ────────────────────────────────────────────────────────
    G = Generator(z_dim=128, img_size=32)
    D = Discriminator(img_size=32)
    print(f"Generator:     {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M params")
    print(f"Discriminator: {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M params")

    # ── Configure ─────────────────────────────────────────────────────
    output_dir = os.path.join(_ROOT, "quickstart_output")

    cfg = TrainerConfig(
        n_epochs=5,                                 # short demo — increase for real training
        batch_size=64,
        lr_g=2e-4,
        lr_d=4e-4,
        ctrl=ControllerConfig(
            gamma_init=10.0,                        # initial GP weight
            viol_target=0.01,                       # target Lipschitz violation
            alpha_gamma=0.01,                       # γ adaptation step size
        ),
        ckpt_dir=output_dir,
        log_every=100,
        save_every=5,
        amp=device.type == "cuda",                  # AMP only on CUDA
    )

    # ── Train ─────────────────────────────────────────────────────────
    trainer = AdaRpGANTrainer(G, D, train_loader, cfg, device, val_loader)
    trainer.fit()

    # ── Post-training plots ───────────────────────────────────────────
    plot_training_curves(
        trainer.log,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )
    plot_controller_state(
        trainer.ctrl.history_viol,
        trainer.ctrl.history_var,
        trainer.ctrl.history_gamma,
        trainer.ctrl.history_n_critic,
        save_path=os.path.join(output_dir, "controller_state.png"),
    )

    print(f"\n✓ Done! Results saved to {output_dir}/")
    print(f"  • Training curves: {output_dir}/training_curves.png")
    print(f"  • Controller state: {output_dir}/controller_state.png")
    print(f"  • Checkpoint: {output_dir}/final.pt")


if __name__ == "__main__":
    main()
