#!/usr/bin/env python3
"""
train.py — Main entry point for AdaRpGAN.

Usage
-----
    # Train AdaRpGAN on CIFAR-10 (default, same as before)
    python train.py

    # Train on CelebA 64×64
    python train.py --dataset celeba --img-size 64

    # Train on custom image folder (FFHQ, CelebA-HQ, etc.)
    python train.py --dataset image_folder --data-dir /path/to/images --img-size 256

    # Train baseline (fixed-γ)
    python train.py --mode baseline --gamma 10.0

    # Resume from checkpoint
    python train.py --resume checkpoints/epoch_0050.pt
"""

import argparse
import os
import sys
import torch

from adarppgan.models  import Generator, Discriminator
from adarppgan.trainer import (
    AdaRpGANTrainer,    TrainerConfig,
    BaselineRpGANTrainer, BaselineConfig,
    ControllerConfig,
)
from adarppgan.utils import (
    get_cifar10_loaders, get_celeba_loader,
    get_image_folder_loader,
    plot_training_curves, plot_controller_state,
)


def parse_args():
    p = argparse.ArgumentParser(description="AdaRpGAN Trainer")

    p.add_argument("--mode",       type=str,   default="adaptive",
                   choices=["adaptive", "baseline"],
                   help="Training mode: adaptive (AdaRpGAN) or baseline (fixed-γ)")

    # Dataset
    p.add_argument("--dataset",    type=str,   default="cifar10",
                   choices=["cifar10", "celeba", "image_folder"],
                   help="Dataset to train on")
    p.add_argument("--img-size",   type=int,   default=32,
                   help="Image resolution (must be power of 2, ≥ 8)")
    p.add_argument("--data-dir",   type=str,   default=None,
                   help="Path to image folder (required for --dataset image_folder)")

    # Architecture
    p.add_argument("--z-dim",      type=int,   default=128)
    p.add_argument("--g-ch",       type=int,   default=256, help="Generator base channels")
    p.add_argument("--d-ch",       type=int,   default=128, help="Discriminator base channels")

    # Optimisation
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr-g",       type=float, default=2e-4)
    p.add_argument("--lr-d",       type=float, default=4e-4)
    p.add_argument("--gp-type",    type=str,   default="wgan-gp",
                   choices=["wgan-gp", "r1"])

    # Adaptive controller
    p.add_argument("--gamma-init", type=float, default=10.0)
    p.add_argument("--gamma-min",  type=float, default=1.0)
    p.add_argument("--gamma-max",  type=float, default=50.0)
    p.add_argument("--alpha-gam",  type=float, default=0.01,
                   help="γ adaptation step size α_γ")
    p.add_argument("--viol-tgt",   type=float, default=0.01,
                   help="Target Lipschitz violation v*")
    p.add_argument("--sigma-hi",   type=float, default=0.50,
                   help="Loss variance threshold to increase n_critic")
    p.add_argument("--sigma-lo",   type=float, default=0.05,
                   help="Loss variance threshold to decrease n_critic")
    p.add_argument("--window",     type=int,   default=50,
                   help="Rolling window width for variance estimation")
    p.add_argument("--cooldown",   type=int,   default=100,
                   help="Cooldown steps after n_critic change")

    # Baseline fixed values
    p.add_argument("--gamma",      type=float, default=10.0,
                   help="Fixed γ for baseline mode")
    p.add_argument("--n-critic",   type=int,   default=1,
                   help="Fixed n_critic for baseline mode")

    # Data / misc
    p.add_argument("--data-root",  type=str,   default="./data")
    p.add_argument("--ckpt-dir",   type=str,   default="./checkpoints")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no-amp",     action="store_true", help="Disable mixed precision")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int,   default=5)
    p.add_argument("--log-every",  type=int,   default=100)

    return p.parse_args()


def build_data_loader(args):
    """Build the appropriate data loader based on --dataset flag."""
    if args.dataset == "cifar10":
        # CIFAR-10 is always 32×32
        train_loader, val_loader = get_cifar10_loaders(
            batch_size  = args.batch,
            data_root   = args.data_root,
            num_workers = args.workers,
        )
        return train_loader, val_loader

    elif args.dataset == "celeba":
        train_loader = get_celeba_loader(
            data_root   = args.data_root,
            img_size    = args.img_size,
            batch_size  = args.batch,
            num_workers = args.workers,
        )
        return train_loader, None

    elif args.dataset == "image_folder":
        assert args.data_dir is not None, \
            "--data-dir is required when --dataset=image_folder"
        train_loader = get_image_folder_loader(
            data_dir    = args.data_dir,
            img_size    = args.img_size,
            batch_size  = args.batch,
            num_workers = args.workers,
        )
        return train_loader, None

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use MPS on Apple Silicon if available and CUDA is not
    if device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"[train.py] device = {device}")
    print(f"[train.py] dataset = {args.dataset}, img_size = {args.img_size}")

    # Reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Enforce img_size for CIFAR-10
    if args.dataset == "cifar10":
        args.img_size = 32

    # Data
    train_loader, val_loader = build_data_loader(args)

    # Models (now resolution-flexible)
    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch, img_size=args.img_size)
    D = Discriminator(base_ch=args.d_ch, img_size=args.img_size)
    n_params_G = sum(p.numel() for p in G.parameters()) / 1e6
    n_params_D = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[Model] G: {n_params_G:.2f}M params  |  D: {n_params_D:.2f}M params")
    print(f"[Model] img_size = {args.img_size}")

    if args.mode == "adaptive":
        ctrl_cfg = ControllerConfig(
            gamma_init    = args.gamma_init,
            gamma_min     = args.gamma_min,
            gamma_max     = args.gamma_max,
            alpha_gamma   = args.alpha_gam,
            viol_target   = args.viol_tgt,
            sigma_high    = args.sigma_hi,
            sigma_low     = args.sigma_lo,
            window        = args.window,
            cooldown      = args.cooldown,
        )
        cfg = TrainerConfig(
            lr_g        = args.lr_g,
            lr_d        = args.lr_d,
            batch_size  = args.batch,
            n_epochs    = args.epochs,
            gp_type     = args.gp_type,
            ctrl        = ctrl_cfg,
            ckpt_dir    = args.ckpt_dir,
            log_every   = args.log_every,
            save_every  = args.save_every,
            amp         = not args.no_amp,
        )
        trainer = AdaRpGANTrainer(G, D, train_loader, cfg, device, val_loader)

        if args.resume:
            trainer.load_checkpoint(args.resume)

        trainer.fit()

        # Post-training plots
        plot_training_curves(
            trainer.log,
            save_path=os.path.join(args.ckpt_dir, "training_curves.png"),
        )
        plot_controller_state(
            history_viol     = trainer.ctrl.history_viol,
            history_var      = trainer.ctrl.history_var,
            history_gamma    = trainer.ctrl.history_gamma,
            history_n_critic = trainer.ctrl.history_n_critic,
            save_path        = os.path.join(args.ckpt_dir, "controller_state.png"),
        )

    else:  # baseline
        cfg = BaselineConfig(
            lr_g        = args.lr_g,
            lr_d        = args.lr_d,
            batch_size  = args.batch,
            n_epochs    = args.epochs,
            gamma       = args.gamma,
            n_critic    = args.n_critic,
            gp_type     = args.gp_type,
            ckpt_dir    = args.ckpt_dir + "_baseline",
            log_every   = args.log_every,
            save_every  = args.save_every,
            amp         = not args.no_amp,
        )
        trainer = BaselineRpGANTrainer(G, D, train_loader, cfg, device)
        trainer.fit()


if __name__ == "__main__":
    main()
