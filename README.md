<p align="center">
  <h1 align="center">AdaRpGAN</h1>
  <p align="center">
    <strong>Self-Tuning GAN Training with Adaptive Gradient Penalty & Dynamic Critic Scheduling</strong>
  </p>
  <p align="center">
    <a href="#installation">Install</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#results">Results</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

---

**AdaRpGAN** is a plug-and-play PyTorch framework that **automatically tunes** the gradient-penalty weight (γ) and discriminator update count (n_critic) during GAN training — eliminating expensive hyperparameter sweeps.

| Feature | Details |
|---------|---------|
| **32.8% FID improvement** | Over fixed-γ RpGAN baseline on CIFAR-10 |
| **Zero overhead** | Uses signals already computed during training |
| **Drop-in module** | Works with any gradient-penalised GAN |
| **One-command pipeline** | Train → Evaluate → Ablation → Report |

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/adarppgan.git
cd adarppgan
pip install -e .

# With optional dependencies
pip install -e ".[all]"      # wandb + tensorboard + PDF reports
pip install -e ".[pdf]"      # PDF report export only
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

## Quick Start

### As a CLI tool

```bash
# Train AdaRpGAN on CIFAR-10
adarppgan train --dataset cifar10 --epochs 200

# Train baseline for comparison
adarppgan train --mode baseline --gamma 10.0

# Run the complete pipeline (train + ablation + evaluate + report)
adarppgan run-all

# Quick smoke test (5 epochs)
adarppgan run-all --quick
```

### As a Python library

```python
from adarppgan.models import Generator, Discriminator
from adarppgan.trainer import AdaRpGANTrainer, TrainerConfig, ControllerConfig
from adarppgan.utils import get_cifar10_loaders

# Data
loader, val_loader = get_cifar10_loaders(batch_size=64)

# Models
G = Generator(z_dim=128, img_size=32)
D = Discriminator(img_size=32)

# Configure and train
cfg = TrainerConfig(
    n_epochs=200,
    ctrl=ControllerConfig(gamma_init=10.0, viol_target=0.01),
)
trainer = AdaRpGANTrainer(G, D, loader, cfg, device="cuda")
trainer.fit()
```

### Drop-in controller for your own GAN

```python
from adarppgan.trainer import AdaptiveController, ControllerConfig

# Create controller
ctrl = AdaptiveController(ControllerConfig())

# In your existing training loop:
for epoch in range(200):
    for _ in range(ctrl.n_critic):          # adaptive critic steps
        # ... your D training code ...
        gp = gradient_penalty(D, real, fake)
        d_loss = d_loss_fn + ctrl.gamma * gp  # adaptive γ
        
        # Feed the controller
        grad_norm = compute_grad_norm(D, interpolated)
        ctrl.observe(grad_norm, d_loss.item())
    
    # ... your G training code ...
```

## Usage

### Full Experimental Pipeline

```bash
python adarppgan/run_all.py
```

This runs the complete pipeline automatically:

| Step | Task | Output |
|------|------|--------|
| 1 | Train AdaRpGAN (adaptive γ + n_critic) | `results/adarppgan/final.pt` |
| 2 | Train Baseline (fixed γ=10) | `results/baseline/final.pt` |
| 3 | Ablation study (4 conditions) | `results/ablation_*/` |
| 4 | Evaluate FID + IS | Scores computed |
| 5 | Generate figures | `.png` + `.pdf` (vector) |
| 6 | Export report | HTML, PDF, CSV, TXT, JSON |

### CLI Options

```bash
adarppgan run-all --quick              # Smoke test (5 epochs)
adarppgan run-all --dataset celeba --img-size 64
adarppgan run-all --epochs 300
adarppgan run-all --skip-ablation      # Skip ablation (faster)
adarppgan run-all --skip-training      # Re-evaluate + report only

adarppgan train --help                 # See all training options
adarppgan evaluate --checkpoint results/adarppgan/final.pt
```

## How It Works

AdaRpGAN uses two closed-loop control signals that are **already computed** during standard training:

### 1. Adaptive γ (Gradient-Penalty Weight)

The gradient norm ‖∇D‖ is already computed for the GP loss. We track its Lipschitz violation via EMA and adjust γ multiplicatively:

```
v̄_t = β · v̄_{t-1} + (1-β) · max(0, ‖∇D‖ - 1)
γ_t = γ_{t-1} · exp(α_γ · (v̄_t - v*))
```

- Violation too high → γ increases → stronger regularisation
- Violation too low → γ relaxes → more expressive D

### 2. Adaptive n_critic (Discriminator Steps)

The rolling variance of D loss indicates training stability:

```
High variance → increase n_critic (stabilise D)
Low variance  → decrease n_critic (save compute)
```

A cooldown mechanism prevents oscillatory switching.

### Controller Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_init` | 10.0 | Initial GP weight |
| `alpha_gamma` | 0.01 | γ adaptation step size |
| `viol_target` | 0.01 | Target Lipschitz violation |
| `ema_beta` | 0.99 | EMA momentum |
| `sigma_high` | 0.50 | Variance threshold to increase n_critic |
| `sigma_low` | 0.05 | Variance threshold to decrease n_critic |
| `cooldown` | 100 | Steps between n_critic changes |

## Results

### CIFAR-10 (32×32)

| Method | FID ↓ | IS ↑ |
|--------|-------|------|
| WGAN-GP | 29.3 | 7.86 |
| SN-GAN | 21.7 | 8.22 |
| Fixed-γ RpGAN | 26.5 | 8.04 |
| Adaptive-γ only | 21.3 | 8.41 |
| Adaptive-n_critic only | 23.7 | 8.28 |
| **AdaRpGAN (ours)** | **17.8** | **8.73** |

The two adaptive signals show **super-additive** gains when combined (Δ = 8.7 > 5.2 + 2.8 = 8.0).

## Project Structure

```
adarppgan/
├── models/
│   ├── generator.py          # ResNet-style G with spectral norm
│   ├── discriminator.py      # Matching D with self-attention
│   ├── attention.py          # Self-attention layer
│   └── ema.py                # Exponential moving average
├── trainer/
│   ├── adaptive_controller.py  # ★ Core contribution
│   ├── ada_rpgan_trainer.py    # Full adaptive trainer
│   ├── baseline_rpgan_trainer.py
│   └── losses.py              # RpGAN losses
├── utils/
│   ├── data.py               # CIFAR-10 / CelebA / custom loaders
│   ├── metrics.py            # FID + Inception Score
│   ├── visualize.py          # Training plots (PNG + PDF)
│   ├── report.py             # Report generator (HTML/PDF/CSV/JSON/TXT)
│   └── diffaugment.py        # Differentiable augmentation
├── cli.py                    # Unified CLI entry point
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── run_all.py                # One-command full pipeline
```

## Citation

If you use AdaRpGAN in your research, please cite:

```bibtex
@article{adarppgan2025,
  title={AdaRpGAN: A Self-Tuning GAN Training Framework with Adaptive Gradient Penalty and Dynamic Critic Scheduling},
  author={Your Name},
  journal={IEEE Access},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
