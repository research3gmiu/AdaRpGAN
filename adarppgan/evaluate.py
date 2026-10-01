#!/usr/bin/env python3
"""
evaluate.py — Compute FID and Inception Score for a trained AdaRpGAN checkpoint.

Usage
-----
    python evaluate.py --checkpoint checkpoints/final.pt
    python evaluate.py --checkpoint checkpoints/final.pt --dataset celeba --img-size 64
    python evaluate.py --checkpoint checkpoints/final.pt --n-samples 50000 --batch 256
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure parent directory is in sys.path when running directly inside adarppgan/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adarppgan.models import Generator, ExponentialMovingAverage
from adarppgan.utils  import (
    get_cifar10_loaders, get_celeba_loader, get_image_folder_loader,
    compute_fid, compute_inception_score,
    generate_report,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AdaRpGAN Checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)

    # Dataset (must match training)
    p.add_argument("--dataset",    type=str, default="cifar10",
                   choices=["cifar10", "celeba", "image_folder"])
    p.add_argument("--img-size",   type=int, default=32)
    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--data-root",  type=str, default="./data")

    # Architecture (must match training)
    p.add_argument("--z-dim",      type=int, default=128)
    p.add_argument("--g-ch",       type=int, default=512)
    p.add_argument("--n-classes",  type=int, default=None,
                   help="Number of classes (default: 10 for cifar10, 0 otherwise)")

    # Evaluation
    p.add_argument("--n-samples",  type=int, default=50000)
    p.add_argument("--batch",      type=int, default=256)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--no-fid",     action="store_true")
    p.add_argument("--no-is",      action="store_true")

    # Report
    p.add_argument("--report",     action="store_true",
                   help="Generate HTML/CSV report")
    p.add_argument("--report-dir", type=str, default="./eval_report")

    return p.parse_args()


def build_loader(args):
    """Build loader matching the training dataset."""
    if args.dataset == "cifar10":
        args.img_size = 32  # enforce
        _, val = get_cifar10_loaders(args.batch, args.data_root, args.workers)
        return val
    elif args.dataset == "celeba":
        return get_celeba_loader(args.data_root, args.img_size, args.batch, args.workers)
    elif args.dataset == "image_folder":
        assert args.data_dir, "--data-dir required for image_folder"
        return get_image_folder_loader(args.data_dir, args.img_size, args.batch, args.workers)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    n_classes = args.n_classes if args.n_classes is not None else (10 if args.dataset == "cifar10" else 0)

    # Load model with correct resolution and class conditioning
    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch, img_size=args.img_size, n_classes=n_classes)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "G_state" in ckpt:
        G.load_state_dict(ckpt["G_state"])
        print(f"Loaded AdaRpGAN checkpoint (epoch {ckpt.get('epoch', '?')})")
    elif "G" in ckpt:
        G.load_state_dict(ckpt["G"])
        print("Loaded baseline checkpoint")
    else:
        G.load_state_dict(ckpt)
        print("Loaded raw state dict")

    G = G.to(device)

    # Apply EMA weights if available
    ema = ExponentialMovingAverage(G)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        ema.apply()
        print("Using EMA weights for evaluation")

    G.eval()

    # Data
    val_loader = build_loader(args)

    # Metrics
    results = {}

    if not args.no_fid:
        fid = compute_fid(G, val_loader, n_samples=args.n_samples,
                          batch_size=args.batch, device=device)
        results["FID"] = fid

    if not args.no_is:
        is_mean, is_std = compute_inception_score(
            G, n_samples=args.n_samples, batch_size=args.batch, device=device
        )
        results["IS_mean"] = is_mean
        results["IS_std"]  = is_std

    # Print results
    print("\n" + "=" * 50)
    print(f"  Evaluation Results — {args.dataset} ({args.img_size}×{args.img_size})")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 50)

    # Optional report
    if args.report:
        report_data = {
            "conditions": [{
                "name": os.path.basename(args.checkpoint),
                "fid": results.get("FID"),
                "is_mean": results.get("IS_mean"),
                "is_std": results.get("IS_std"),
                "epochs": ckpt.get("epoch", "?"),
            }],
            "dataset": args.dataset.upper(),
            "img_size": args.img_size,
            "device": str(device),
            "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
            "figures": {},
        }
        generate_report(report_data, output_dir=args.report_dir,
                        title=f"AdaRpGAN Evaluation — {args.dataset}")

    # Restore original weights
    if "ema" in ckpt:
        ema.restore()


if __name__ == "__main__":
    main()
