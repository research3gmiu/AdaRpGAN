"""
AdaRpGAN — Adaptive Regularized Relativistic GAN Trainer
=========================================================

A self-contained PyTorch module that jointly adapts:
  γ  (gradient-penalty weight)  via Lipschitz violation EMA
  n_critic (D steps per G step) via rolling discriminator loss variance

Usage
-----
    from models  import Generator, Discriminator
    from trainer import AdaRpGANTrainer, TrainerConfig, ControllerConfig
    from utils   import get_cifar10_loaders

    train_loader, _ = get_cifar10_loaders(batch_size=64)
    G, D = Generator(), Discriminator()
    trainer = AdaRpGANTrainer(G, D, train_loader, TrainerConfig(), device)
    trainer.fit()

Project layout
--------------
    models/
        generator.py        ResNet-style upsampling generator
        discriminator.py    ResNet-style downsampling discriminator + GP utils
    trainer/
        adaptive_controller.py   ← core AdaRpGAN contribution
        ada_rpgan_trainer.py     full training loop (adaptive)
        baseline_rpgan_trainer.py fixed-γ baseline
        losses.py                RpGAN / vanilla GAN losses
    utils/
        data.py             CIFAR-10 DataLoader helpers
        metrics.py          FID + Inception Score
        visualize.py        training curve + controller-state plots
    configs/
        cifar10.yaml        default experiment config
    train.py                CLI entry point
    evaluate.py             FID / IS evaluation
    ablation.py             4-condition ablation runner
"""

__version__ = "1.0.0"
__author__  = "AdaRpGAN Authors"
