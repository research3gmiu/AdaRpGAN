#!/usr/bin/env python3
"""
cli.py — Unified CLI entry point for AdaRpGAN.

After ``pip install -e .``, provides the ``adarppgan`` command with subcommands:

    adarppgan train       [ARGS]    Train AdaRpGAN or baseline
    adarppgan evaluate    [ARGS]    Evaluate a saved checkpoint (FID / IS)
    adarppgan run-all     [ARGS]    Run the full experimental pipeline

Each subcommand forwards its arguments to the matching module's ``main()``.

Examples
--------
    adarppgan train --dataset cifar10 --epochs 200
    adarppgan evaluate --checkpoint results/adarppgan/final.pt
    adarppgan run-all --quick
    adarppgan train --help
"""

import sys


def main():
    """Top-level CLI dispatcher for the ``adarppgan`` console script."""

    # ── show top-level help when called bare ──────────────────────────
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return

    subcommand = sys.argv[1]

    # Strip the subcommand from argv so each module's own argparse works
    # e.g.  adarppgan train --epochs 200  →  sys.argv becomes [train, --epochs, 200]
    sys.argv = [subcommand] + sys.argv[2:]

    if subcommand == "train":
        from adarppgan.train import main as train_main
        train_main()

    elif subcommand == "evaluate":
        from adarppgan.evaluate import main as evaluate_main
        evaluate_main()

    elif subcommand in ("run-all", "run_all"):
        from adarppgan.run_all import main as run_all_main
        run_all_main()

    elif subcommand == "version":
        from adarppgan import __version__
        print(f"adarppgan {__version__}")

    else:
        print(f"Error: unknown subcommand '{subcommand}'\n")
        _print_help()
        sys.exit(1)


def _print_help():
    """Print the top-level usage message."""
    help_text = """\
usage: adarppgan <command> [options]

AdaRpGAN — Self-Tuning GAN Training Framework

commands:
  train          Train AdaRpGAN (adaptive) or baseline (fixed-γ)
  evaluate       Evaluate a saved checkpoint (FID + Inception Score)
  run-all        Run the complete experimental pipeline
  version        Show version number

examples:
  adarppgan train --dataset cifar10 --epochs 200
  adarppgan train --mode baseline --gamma 10.0
  adarppgan evaluate --checkpoint results/adarppgan/final.pt
  adarppgan run-all --quick
  adarppgan train --help          (show train-specific options)

For more info, see: https://github.com/yourusername/adarppgan
"""
    print(help_text)


if __name__ == "__main__":
    main()
