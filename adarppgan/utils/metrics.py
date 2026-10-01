"""
Evaluation metrics: FID and Inception Score.

Both metrics use InceptionV3 features.

FID  (Fréchet Inception Distance) — lower is better.
IS   (Inception Score)            — higher is better.

Usage
-----
    from utils.metrics import compute_fid, compute_inception_score

    fid = compute_fid(G, real_loader, n_samples=50000, device=device)
    IS_mean, IS_std = compute_inception_score(G, n_samples=50000, device=device)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms


# ──────────────────────────────────────────────
# InceptionV3 Feature Extractor
# ──────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """
    InceptionV3 with the classification head removed.
    Returns 2048-dim pool3 features.
    Input: (B, 3, H, W) in [0, 1], H/W ≥ 75 (will resize if needed).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()         # remove classification head
        model.aux_logits = False
        self.model = model.eval().to(device)
        self.device = device
        self.resize = transforms.Resize((299, 299), antialias=True)

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) in [-1, 1]  → (B, 2048) features."""
        imgs = (imgs + 1.0) / 2.0          # [-1,1] → [0,1]
        imgs = self.resize(imgs)
        return self.model(imgs)


# ──────────────────────────────────────────────
# FID
# ──────────────────────────────────────────────

def _get_stats(
    feat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu  = feat.mean(axis=0)
    cov = np.cov(feat, rowvar=False)
    return mu, cov


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Numpy FID calculation between two distributions."""
    diff = mu1 - mu2
    # Numerically stable sqrt of matrix product
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset    = np.eye(sigma1.shape[0]) * eps
        covmean   = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def extract_features(
    G,
    extractor: InceptionFeatureExtractor,
    n_samples: int,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Generate n_samples fakes and extract Inception features."""
    feats = []
    n_done = 0
    G.eval()
    while n_done < n_samples:
        bs   = min(batch_size, n_samples - n_done)
        imgs = G.sample(bs, device)
        f    = extractor(imgs).cpu().numpy()
        feats.append(f)
        n_done += bs
    G.train()
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def extract_real_features(
    loader,
    extractor: InceptionFeatureExtractor,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    feats  = []
    n_done = 0
    for batch in loader:
        if n_done >= n_samples:
            break
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        f    = extractor(imgs).cpu().numpy()
        feats.append(f)
        n_done += imgs.size(0)
    return np.concatenate(feats, axis=0)[:n_samples]


def compute_fid(
    G,
    real_loader,
    n_samples: int = 50000,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute FID between G-generated images and real_loader."""
    ext = InceptionFeatureExtractor(device)
    print(f"[FID] Extracting real features ({n_samples:,} samples)…")
    real_feats = extract_real_features(real_loader, ext, n_samples, device)
    print(f"[FID] Extracting fake features…")
    fake_feats = extract_features(G, ext, n_samples, batch_size, device)

    mu_r, cov_r = _get_stats(real_feats)
    mu_f, cov_f = _get_stats(fake_feats)
    fid = frechet_distance(mu_r, cov_r, mu_f, cov_f)
    print(f"[FID] = {fid:.3f}")
    return fid


# ──────────────────────────────────────────────
# Inception Score
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_inception_score(
    G,
    n_samples:  int = 50000,
    batch_size: int = 256,
    splits:     int = 10,
    device:     torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Compute Inception Score (mean ± std over `splits` splits).
    Returns (IS_mean, IS_std).
    """
    inception = inception_v3(pretrained=True, transform_input=False).eval().to(device)
    resize    = transforms.Resize((299, 299), antialias=True)
    softmax   = nn.Softmax(dim=1)

    probs = []
    n_done = 0
    G.eval()
    while n_done < n_samples:
        bs   = min(batch_size, n_samples - n_done)
        imgs = G.sample(bs, device)
        imgs = (imgs + 1.0) / 2.0
        imgs = resize(imgs)
        p    = softmax(inception(imgs)).cpu().numpy()
        probs.append(p)
        n_done += bs
    G.train()

    probs = np.concatenate(probs, axis=0)
    scores = []
    chunk  = n_samples // splits
    for i in range(splits):
        p_chunk = probs[i * chunk: (i + 1) * chunk]
        p_y     = p_chunk.mean(axis=0, keepdims=True)
        kl      = p_chunk * (np.log(p_chunk + 1e-10) - np.log(p_y + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))
