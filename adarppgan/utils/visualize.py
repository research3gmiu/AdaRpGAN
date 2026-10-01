"""
Visualization and logging utilities for AdaRpGAN.

plot_training_curves()   — loss curves + γ + n_critic over time
plot_controller_state()  — Lipschitz violation EMA + loss variance
plot_sample_grid()       — 8×8 grid of generated samples

All plots are saved as both PNG (raster, 150 dpi) and PDF (vector, paper-ready).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


# ──────────────────────────────────────────────
# Training Curves
# ──────────────────────────────────────────────

def plot_training_curves(
    log: dict,
    save_path: str = "training_curves.png",
    title:     str = "AdaRpGAN — Training Curves",
) -> None:
    """
    Plot D/G loss, γ, and n_critic over training epochs.

    log keys expected: 'd_loss', 'g_loss', 'gamma', 'n_critic'
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = np.arange(1, len(log.get("d_loss", [])) + 1)

    # D loss
    ax = axes[0, 0]
    ax.plot(epochs, log.get("d_loss", []), color="#e74c3c", linewidth=1.5, label="D loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Discriminator Loss"); ax.legend(); ax.grid(alpha=0.3)

    # G loss
    ax = axes[0, 1]
    ax.plot(epochs, log.get("g_loss", []), color="#3498db", linewidth=1.5, label="G loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Generator Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Gamma
    ax = axes[1, 0]
    gamma_log = log.get("gamma", [])
    if len(gamma_log) == len(epochs):
        ax.plot(epochs, gamma_log, color="#2ecc71", linewidth=1.5, label="γ (adaptive)")
    if "gamma_baseline" in log:
        ax.axhline(log["gamma_baseline"], color="gray", linestyle="--", label="γ (fixed baseline)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("γ")
    ax.set_title("Adaptive GP Weight γ"); ax.legend(); ax.grid(alpha=0.3)

    # n_critic
    ax = axes[1, 1]
    n_c = log.get("n_critic", [])
    if len(n_c) == len(epochs):
        ax.step(epochs, n_c, color="#9b59b6", linewidth=1.5, where="post", label="n_critic")
    ax.set_xlabel("Epoch"); ax.set_ylabel("n_critic")
    ax.set_title("Adaptive n_critic"); ax.set_yticks(range(1, 6))
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # Also save as PDF for paper-ready vector figures
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {save_path}  (+PDF)")


def plot_controller_state(
    history_viol:     list,
    history_var:      list,
    history_gamma:    list,
    history_n_critic: list,
    save_path: str = "controller_state.png",
) -> None:
    """
    Plot the internal state of the adaptive controller over D-steps.
    """
    steps = np.arange(len(history_viol))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("AdaptiveController Internal State", fontsize=13, fontweight="bold")

    pairs = [
        (axes[0, 0], history_viol,     "#e67e22", "Lipschitz Violation EMA v̄",      "v̄"),
        (axes[0, 1], history_var,      "#e74c3c", "D Loss Variance (rolling)",        "σ²"),
        (axes[1, 0], history_gamma,    "#2ecc71", "γ  (GP weight)",                   "γ"),
        (axes[1, 1], history_n_critic, "#9b59b6", "n_critic",                         "n_critic"),
    ]

    for ax, data, color, title, ylabel in pairs:
        ax.plot(steps, data, color=color, linewidth=1.0, alpha=0.9)
        ax.set_title(title); ax.set_xlabel("D step"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {save_path}  (+PDF)")


def plot_fid_comparison(
    fid_ada:      list,
    fid_baseline: list,
    epochs:       list,
    save_path:    str = "fid_comparison.png",
) -> None:
    """Bar/line chart comparing FID curves of AdaRpGAN vs baseline."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, fid_ada,      "o-", color="#2ecc71", linewidth=2,
            markersize=5, label="AdaRpGAN (ours)")
    ax.plot(epochs, fid_baseline, "s--", color="#e74c3c", linewidth=2,
            markersize=5, label="Baseline RpGAN (fixed-γ)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID ↓", fontsize=12)
    ax.set_title("FID Comparison: AdaRpGAN vs Fixed-γ Baseline", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()      # Lower FID is better — show improvements going "up"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {save_path}  (+PDF)")


def plot_ablation_bar(
    names:    list[str],
    fids:     list[float],
    save_path: str = "ablation.png",
) -> None:
    """
    Horizontal bar chart for ablation study.
    names: ['AdaRpGAN (full)', 'Adaptive-γ only', 'Adaptive-n_critic only', 'Fixed-γ baseline']
    """
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, fids, color=colors[:len(names)], edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, fids):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=10)

    ax.set_xlabel("FID ↓ (lower is better)", fontsize=11)
    ax.set_title("Ablation Study on CIFAR-10", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved → {save_path}  (+PDF)")
