# AdaRpGAN Codebase Understanding

This document summarizes the architecture, core components, and functionalities of the AdaRpGAN project.

## 1. Project Overview
**AdaRpGAN** is an Adaptive Regularized Relativistic GAN Trainer built in PyTorch. The main innovation of this codebase is the dynamic tuning of GAN training hyperparameters that are traditionally difficult to set manually:
*   **Gradient-Penalty Weight ($\gamma$)**: Dynamically adjusted based on the discriminator's Lipschitz violation constraint in real-time.
*   **Number of Critic Steps ($n_{critic}$)**: Adjusted up or down depending on the rolling variance of the discriminator's loss.

This approach eliminates the need for exhaustive hyperparameter search and improves convergence speeds, leading to better Fréchet Inception Distance (FID) and Inception Scores (IS) on the CIFAR-10 dataset.

## 2. Core Architecture and Modules

### A. The Adaptive Controller (`trainer/adaptive_controller.py`)
This is the heart of the project.
*   **Lipschitz Violation Tracking**: The controller monitors the mean gradient norm `||∇D(x̂)||₂`. It computes the violation `v_t = max(0, norm_t - 1)` and maintains an Exponential Moving Average (EMA). The parameter $\gamma$ is then smoothly scaled to match a target violation level.
*   **Loss Variance Tracking**: The controller maintains a rolling window (default width=50) of the discriminator loss. If the loss variance exceeds a threshold (`σ_high`), it increases the number of discriminator steps per generator step ($n_{critic}$). If the variance falls below a threshold (`σ_low`), it decreases $n_{critic}$.

### B. Trainer Implementation (`trainer/ada_rpgan_trainer.py`)
The `AdaRpGANTrainer` handles the overall training lifecycle.
*   **Step Logic**: A single call to `train_step()` natively loops through $n_{critic}$ discriminator updates followed by a single generator update.
*   **Relativistic GAN (RpGAN) Losses**: Unlike standard GANs which classify absolute real/fake probabilities, RpGAN predicts *relative realness*. The losses are defined in `trainer/losses.py`.
*   **Mixed Precision**: The trainer supports Automatic Mixed Precision (AMP) via PyTorch's `GradScaler` for faster training on modern GPUs.
*   **Stateful Resumption**: The trainer's `save_checkpoint` method exports the neural network weights *and* the internal state of the `AdaptiveController` (including its buffer history and current $\gamma$/$n_{critic}$), allowing interrupted training sessions to resume perfectly without breaking the dynamic adaptations.

### C. Neural Network Models (`models/`)
*   **Generator (`generator.py`)**: A DCGAN-like convolutional network mapping a 128-dimensional latent vector $z$ to a $32 \times 32$ RGB image. It relies heavily on `ResBlockUp` modules which use explicit bilinear upsampling instead of transposed convolutions to avoid checkerboard artifacts. It utilizes Spectral Normalization and Conditional Batch Normalization.
*   **Discriminator (`discriminator.py`)**: A ResNet-style SN-GAN architecture built with `ResBlockDown` blocks utilizing average pooling. Crucially, the final head outputs raw, un-activated logits required for the relativistic loss functions. The file also includes utility wrappers to compute WGAN-GP (two-sided) and R1 (one-sided) regularization metrics.

### D. Utilities (`utils/`)
*   **Metrics (`metrics.py`)**: Responsible for quantitative evaluation using the standard GAN metrics: Fréchet Inception Distance (FID) and Inception Score (IS). This is done through a modified `InceptionV3` network instance loaded via `torchvision`, which passes batched images and extracts `pool3` features to perform matrix evaluations.
*   **Data & Visualization**: Auxiliary files handles CIFAR-10 downloading/loading as well as visualizing generated image grids and plotting the adaptive controller's parameter curves over time.

## 3. Data Flow during Execution
1. Execution starts at `train.py`.
2. The `Generator` and `Discriminator` are instantiated.
3. The `AdaptiveController` config is built.
4. The `AdaRpGANTrainer` orchestrates the process, making decisions on how many times to execute the `Discriminator` backward pass before the `Generator` updates, guided directly by the controller.
5. Intermediate states and synthetic images are periodically dumped into a structured check-pointing system.
