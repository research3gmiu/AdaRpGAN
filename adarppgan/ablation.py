#!/usr/bin/env python3
"""
ablation.py — Run the 4-condition ablation study.

Conditions
----------
  A  AdaRpGAN (full: adaptive γ + adaptive n_critic)
  B  Adaptive-γ only  (n_critic fixed=1)
  C  Adaptive-n_critic only  (γ fixed=10)
  D  Baseline  (γ fixed=10, n_critic fixed=1)

Each condition trains for --epochs epochs on CIFAR-10 and reports FID.
Results are saved to ablation_results.csv and plotted as ablation.png.

Usage
-----
    python ablation.py --epochs 100 --batch 64
"""

import argparse
import os
import sys
import csv
import torch

from adarppgan.models  import Generator, Discriminator
from adarppgan.trainer import (
    AdaRpGANTrainer,    TrainerConfig,
    BaselineRpGANTrainer, BaselineConfig,
    ControllerConfig,
)
from adarppgan.utils import get_cifar10_loaders, compute_fid, plot_ablation_bar


def make_ada_trainer(G, D, loader, device, epochs, ckpt_dir,
                     adapt_gamma=True, adapt_ncritic=True):
    ctrl = ControllerConfig(
        alpha_gamma  = 0.01  if adapt_gamma   else 0.0,   # α=0 → γ frozen
        sigma_high   = 0.50  if adapt_ncritic else 1e9,   # σ_hi=∞ → n_critic frozen
        sigma_low    = 0.05  if adapt_ncritic else -1e9,
    )
    cfg = TrainerConfig(
        n_epochs=epochs, ckpt_dir=ckpt_dir, ctrl=ctrl,
        log_every=200, save_every=9999, sample_every=9999,
    )
    return AdaRpGANTrainer(G, D, loader, cfg, device)


def make_baseline_trainer(G, D, loader, device, epochs, ckpt_dir):
    cfg = BaselineConfig(
        n_epochs=epochs, ckpt_dir=ckpt_dir,
        log_every=200, save_every=9999,
    )
    return BaselineRpGANTrainer(G, D, loader, cfg, device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--batch",     type=int, default=64)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir",   type=str, default="./ablation_results")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    loader, val_loader = get_cifar10_loaders(
        batch_size=args.batch, data_root=args.data_root
    )

    conditions = [
        ("AdaRpGAN (full)",            True,  True),
        ("Adaptive-γ only",            True,  False),
        ("Adaptive-n_critic only",     False, True),
        ("Fixed-γ baseline",           False, False),
    ]

    results = []

    for name, adapt_g, adapt_n in conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {name}")
        print(f"{'='*60}")

        G = Generator(); D = Discriminator()
        ckpt_dir = os.path.join(args.out_dir, name.replace(" ", "_").replace("(", "").replace(")", ""))

        if adapt_g or adapt_n:
            trainer = make_ada_trainer(
                G, D, loader, device, args.epochs, ckpt_dir,
                adapt_gamma=adapt_g, adapt_ncritic=adapt_n,
            )
        else:
            trainer = make_baseline_trainer(G, D, loader, device, args.epochs, ckpt_dir)

        trainer.fit()

        print(f"\n[Ablation] Computing FID for '{name}'…")
        fid = compute_fid(G, val_loader, n_samples=10000,
                          batch_size=256, device=device)
        results.append((name, fid))
        print(f"  → FID = {fid:.3f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition", "FID"])
        w.writerows(results)
    print(f"\n[Ablation] Results saved → {csv_path}")

    # Print table
    print("\n" + "="*45)
    print(f"{'Condition':<35} {'FID':>8}")
    print("-"*45)
    for name, fid in results:
        print(f"{name:<35} {fid:>8.3f}")
    print("="*45)

    # Bar plot
    names = [r[0] for r in results]
    fids  = [r[1] for r in results]
    plot_ablation_bar(names, fids,
                      save_path=os.path.join(args.out_dir, "ablation.png"))


if __name__ == "__main__":
    main()
