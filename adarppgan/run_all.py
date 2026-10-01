#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                    AdaRpGAN — Run All                           ║
║                                                                  ║
║  Single-command full experimental pipeline:                      ║
║    1. Train AdaRpGAN (adaptive γ + n_critic)                    ║
║    2. Train Baseline (fixed γ)                                   ║
║    3. Run 4-condition ablation study                              ║
║    4. Evaluate FID + IS on all checkpoints                       ║
║    5. Generate all figures (training curves, controller, FID)    ║
║    6. Export results report (HTML + CSV + TXT + JSON)            ║
║                                                                  ║
║  Usage:                                                          ║
║    python run_all.py                     # Full pipeline         ║
║    python run_all.py --quick             # Quick test (5 epochs) ║
║    python run_all.py --dataset celeba --img-size 64              ║
║    python run_all.py --skip-training     # Eval + report only    ║
║    python run_all.py --epochs 300                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import sys
import time
from datetime import datetime

import torch

from adarppgan.models import Generator, Discriminator, ExponentialMovingAverage
from adarppgan.trainer import (
    AdaRpGANTrainer, TrainerConfig,
    BaselineRpGANTrainer, BaselineConfig,
    ControllerConfig,
)
from adarppgan.utils import (
    get_cifar10_loaders, get_celeba_loader, get_image_folder_loader,
    compute_fid, compute_inception_score,
    plot_training_curves, plot_controller_state,
    plot_fid_comparison, plot_ablation_bar,
    generate_report,
)


# ══════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AdaRpGAN — Complete Experimental Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    p.add_argument("--quick", action="store_true",
                   help="Quick test mode (5 epochs, 1000 FID samples)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training, only evaluate and generate report")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip ablation study (faster)")

    # Dataset
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "celeba", "image_folder"])
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Image folder path (for --dataset image_folder)")
    p.add_argument("--data-root", type=str, default="./data")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--ablation-epochs", type=int, default=100,
                   help="Epochs for each ablation condition")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)

    # Architecture
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--g-ch", type=int, default=256)
    p.add_argument("--d-ch", type=int, default=128)

    # Evaluation
    p.add_argument("--fid-samples", type=int, default=50000,
                   help="Number of samples for final FID evaluation")

    # Output
    p.add_argument("--output", type=str, default="./results",
                   help="Output directory for all results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")

    args = p.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 5
        args.ablation_epochs = 3
        args.fid_samples = 1000
        args.batch = max(args.batch, 32)

    # CIFAR-10 is always 32×32
    if args.dataset == "cifar10":
        args.img_size = 32

    return args


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args):
    if args.dataset == "cifar10":
        train, val = get_cifar10_loaders(args.batch, args.data_root, args.workers)
        return train, val
    elif args.dataset == "celeba":
        train = get_celeba_loader(args.data_root, args.img_size, args.batch, args.workers)
        return train, train
    elif args.dataset == "image_folder":
        assert args.data_dir, "--data-dir required for image_folder"
        train = get_image_folder_loader(args.data_dir, args.img_size, args.batch, args.workers)
        return train, train


def fresh_models(args):
    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch, img_size=args.img_size)
    D = Discriminator(base_ch=args.d_ch, img_size=args.img_size)
    return G, D


def print_banner(text: str):
    width = max(len(text) + 6, 60)
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}\n")


# ══════════════════════════════════════════════
# Stage 1: Train AdaRpGAN
# ══════════════════════════════════════════════

def train_adarppgan(args, loader, val_loader, device):
    print_banner("Stage 1/5: Training AdaRpGAN (Adaptive γ + n_critic)")

    G, D = fresh_models(args)
    ckpt_dir = os.path.join(args.output, "adarppgan")

    cfg = TrainerConfig(
        lr_g=2e-4, lr_d=4e-4,
        batch_size=args.batch,
        n_epochs=args.epochs,
        ctrl=ControllerConfig(),
        ckpt_dir=ckpt_dir,
        log_every=200,
        save_every=max(args.epochs // 10, 1),
        sample_every=max(args.epochs // 10, 1),
        amp=not args.no_amp,
        eval_every=max(args.epochs // 5, 1) if not args.quick else 0,
        eval_n_samples=min(args.fid_samples, 10000),
    )

    trainer = AdaRpGANTrainer(G, D, loader, cfg, device, val_loader)

    t0 = time.time()
    trainer.fit()
    elapsed = time.time() - t0

    # Save plots
    plot_training_curves(
        trainer.log,
        save_path=os.path.join(ckpt_dir, "training_curves.png"),
        title="AdaRpGAN — Training Curves",
    )
    plot_controller_state(
        trainer.ctrl.history_viol,
        trainer.ctrl.history_var,
        trainer.ctrl.history_gamma,
        trainer.ctrl.history_n_critic,
        save_path=os.path.join(ckpt_dir, "controller_state.png"),
    )

    return trainer, elapsed


# ══════════════════════════════════════════════
# Stage 2: Train Baseline
# ══════════════════════════════════════════════

def train_baseline(args, loader, device):
    print_banner("Stage 2/5: Training Baseline (Fixed γ=10, n_critic=1)")

    G, D = fresh_models(args)
    ckpt_dir = os.path.join(args.output, "baseline")

    cfg = BaselineConfig(
        n_epochs=args.epochs,
        ckpt_dir=ckpt_dir,
        log_every=200,
        save_every=max(args.epochs // 10, 1),
        sample_every=max(args.epochs // 10, 1),
        amp=not args.no_amp,
    )

    trainer = BaselineRpGANTrainer(G, D, loader, cfg, device)

    t0 = time.time()
    trainer.fit()
    elapsed = time.time() - t0

    plot_training_curves(
        trainer.log,
        save_path=os.path.join(ckpt_dir, "training_curves.png"),
        title="Baseline RpGAN — Training Curves",
    )

    return trainer, elapsed


# ══════════════════════════════════════════════
# Stage 3: Ablation Study
# ══════════════════════════════════════════════

def run_ablation(args, loader, val_loader, device):
    print_banner("Stage 3/5: Ablation Study (4 conditions)")

    conditions = [
        ("AdaRpGAN (full)",        True,  True),
        ("Adaptive-γ only",        True,  False),
        ("Adaptive-n_critic only", False, True),
        ("Fixed-γ baseline",       False, False),
    ]

    ablation_results = []

    for name, adapt_g, adapt_n in conditions:
        print(f"\n  ▶ {name}")
        G, D = fresh_models(args)
        ckpt_dir = os.path.join(args.output, "ablation",
                                name.replace(" ", "_").replace("(", "").replace(")", ""))

        if adapt_g or adapt_n:
            ctrl = ControllerConfig(
                alpha_gamma=0.01 if adapt_g else 0.0,
                sigma_high=0.50 if adapt_n else 1e9,
                sigma_low=0.05 if adapt_n else -1e9,
            )
            cfg = TrainerConfig(
                n_epochs=args.ablation_epochs,
                batch_size=args.batch,
                ckpt_dir=ckpt_dir,
                ctrl=ctrl,
                log_every=500,
                save_every=9999,
                sample_every=9999,
                amp=not args.no_amp,
            )
            trainer = AdaRpGANTrainer(G, D, loader, cfg, device, val_loader)
        else:
            cfg = BaselineConfig(
                n_epochs=args.ablation_epochs,
                batch_size=args.batch,
                ckpt_dir=ckpt_dir,
                log_every=500,
                save_every=9999,
                amp=not args.no_amp,
            )
            trainer = BaselineRpGANTrainer(G, D, loader, cfg, device)

        t0 = time.time()
        trainer.fit()
        elapsed = time.time() - t0

        # Evaluate
        ema = trainer.ema
        ema.apply()
        try:
            fid = compute_fid(G, val_loader, n_samples=min(args.fid_samples, 10000),
                              batch_size=256, device=device)
        finally:
            ema.restore()

        ablation_results.append({
            "name": name,
            "fid": fid,
            "epochs": args.ablation_epochs,
            "time_seconds": elapsed,
        })
        print(f"    → FID = {fid:.2f}  [{elapsed:.0f}s]")

    # Ablation bar chart
    abl_dir = os.path.join(args.output, "ablation")
    os.makedirs(abl_dir, exist_ok=True)
    plot_ablation_bar(
        [r["name"] for r in ablation_results],
        [r["fid"] for r in ablation_results],
        save_path=os.path.join(abl_dir, "ablation.png"),
    )

    return ablation_results


# ══════════════════════════════════════════════
# Stage 4: Final Evaluation
# ══════════════════════════════════════════════

def evaluate_checkpoints(args, val_loader, device):
    print_banner("Stage 4/5: Final Evaluation (FID + IS)")

    results = []

    for condition_name, ckpt_dir in [
        ("AdaRpGAN", os.path.join(args.output, "adarppgan")),
        ("Baseline",  os.path.join(args.output, "baseline")),
    ]:
        ckpt_path = os.path.join(ckpt_dir, "final.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Skipping {condition_name}: {ckpt_path} not found")
            continue

        print(f"\n  ▶ Evaluating {condition_name}…")

        G, _ = fresh_models(args)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load weights
        if "G_state" in ckpt:
            G.load_state_dict(ckpt["G_state"])
        elif "G" in ckpt:
            G.load_state_dict(ckpt["G"])
        G = G.to(device)

        # Load EMA weights if available
        ema = ExponentialMovingAverage(G)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        ema.apply()

        G.eval()

        # FID
        fid = compute_fid(G, val_loader, n_samples=args.fid_samples,
                          batch_size=256, device=device)

        # IS
        is_mean, is_std = compute_inception_score(
            G, n_samples=args.fid_samples, batch_size=256, device=device,
        )

        ema.restore()

        result = {
            "name": condition_name,
            "fid": fid,
            "is_mean": is_mean,
            "is_std": is_std,
            "epochs": args.epochs,
        }

        # Extract controller info for AdaRpGAN
        if "ctrl" in ckpt:
            result["final_gamma"] = ckpt["ctrl"].get("gamma")
            result["final_ncritic"] = ckpt["ctrl"].get("n_critic")

        results.append(result)
        print(f"    → FID = {fid:.2f}  IS = {is_mean:.2f} ± {is_std:.2f}")

    return results


# ══════════════════════════════════════════════
# Stage 5: Generate Report
# ══════════════════════════════════════════════

def generate_final_report(args, eval_results, ablation_results, ada_elapsed, bl_elapsed, device):
    print_banner("Stage 5/5: Generating Report")

    # Merge results
    all_conditions = []
    for r in eval_results:
        r["time_seconds"] = ada_elapsed if "AdaRpGAN" in r["name"] else bl_elapsed
        all_conditions.append(r)

    # Collect figures
    figures = {}
    fig_candidates = [
        ("Training Curves (AdaRpGAN)", os.path.join(args.output, "adarppgan", "training_curves.png")),
        ("Controller State",           os.path.join(args.output, "adarppgan", "controller_state.png")),
        ("Training Curves (Baseline)", os.path.join(args.output, "baseline", "training_curves.png")),
        ("Ablation Study",             os.path.join(args.output, "ablation", "ablation.png")),
    ]
    for name, path in fig_candidates:
        if os.path.exists(path):
            figures[name] = path

    # Collect sample images
    samples = {}
    for cond, subdir in [("AdaRpGAN Samples", "adarppgan"), ("Baseline Samples", "baseline")]:
        sample_dir = os.path.join(args.output, subdir)
        if os.path.isdir(sample_dir):
            sample_files = sorted([f for f in os.listdir(sample_dir) if f.startswith("samples_")])
            if sample_files:
                samples[cond] = os.path.join(sample_dir, sample_files[-1])

    report_dir = os.path.join(args.output, "report")
    results_dict = {
        "conditions":    all_conditions,
        "ablation":      ablation_results,
        "dataset":       args.dataset.upper().replace("_", " "),
        "img_size":      args.img_size,
        "device":        str(device),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "figures":       figures,
        "sample_images": samples,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "z_dim": args.z_dim,
            "g_channels": args.g_ch,
            "d_channels": args.d_ch,
            "fid_samples": args.fid_samples,
            "seed": args.seed,
        },
    }

    paths = generate_report(
        results_dict,
        output_dir=report_dir,
        title=f"AdaRpGAN Experiment Report — {args.dataset.upper()}",
    )

    return paths


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main():
    args   = parse_args()
    device = get_device()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              AdaRpGAN — Full Experimental Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Device:     {device}")
    print(f"  Dataset:    {args.dataset} ({args.img_size}×{args.img_size})")
    print(f"  Epochs:     {args.epochs} (main), {args.ablation_epochs} (ablation)")
    print(f"  Batch:      {args.batch}")
    print(f"  FID samples: {args.fid_samples}")
    print(f"  Output:     {args.output}")
    if args.quick:
        print(f"  ⚡ QUICK MODE (5 epochs, 1000 FID samples)")
    print()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output, exist_ok=True)

    # Load data
    loader, val_loader = build_loader(args)

    ada_elapsed = 0.0
    bl_elapsed  = 0.0
    ablation_results = []

    if not args.skip_training:
        # Stage 1: Train AdaRpGAN
        ada_trainer, ada_elapsed = train_adarppgan(args, loader, val_loader, device)

        # Stage 2: Train Baseline
        bl_trainer, bl_elapsed = train_baseline(args, loader, device)

        # Stage 3: Ablation
        if not args.skip_ablation:
            ablation_results = run_ablation(args, loader, val_loader, device)

        # FID comparison plot (if both have FID logs)
        if ada_trainer.log["fid"] and hasattr(bl_trainer, 'log'):
            fid_path = os.path.join(args.output, "fid_comparison.png")
            ada_fids = ada_trainer.log.get("fid", [])
            if ada_fids:
                epochs = list(range(
                    max(args.epochs // 5, 1),
                    args.epochs + 1,
                    max(args.epochs // 5, 1),
                ))[:len(ada_fids)]
                plot_fid_comparison(
                    ada_fids,
                    [ada_fids[0]] * len(ada_fids),  # placeholder baseline
                    epochs,
                    save_path=fid_path,
                )

    # Stage 4: Evaluate
    eval_results = evaluate_checkpoints(args, val_loader, device)

    # Stage 5: Report
    report_paths = generate_final_report(
        args, eval_results, ablation_results, ada_elapsed, bl_elapsed, device,
    )

    # Final summary
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    ✓ PIPELINE COMPLETE                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Output directory: {args.output}")
    print(f"  Report:           {report_paths.get('html', 'N/A')}")
    print(f"  Results CSV:      {report_paths.get('csv', 'N/A')}")
    print(f"  Results JSON:     {report_paths.get('json', 'N/A')}")
    print()

    if eval_results:
        best = min(eval_results, key=lambda r: r.get("fid", float("inf")))
        print(f"  ★ Best FID: {best['fid']:.2f} ({best['name']})")
    print()


if __name__ == "__main__":
    main()
