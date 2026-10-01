"""
Exponential Moving Average (EMA) of Generator weights.

Standard technique since ProGAN/StyleGAN — uses a shadow copy of G's
parameters for evaluation and sampling, which dramatically smooths FID.

Usage
-----
    ema = ExponentialMovingAverage(G, decay=0.9999)

    # Inside training loop, after G optimizer step:
    ema.update()

    # For evaluation / sampling:
    ema.apply()        # swap EMA weights into G
    fid = compute_fid(G, ...)
    ema.restore()      # swap original weights back
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of a model's parameters.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to track (typically the Generator).
    decay : float
        EMA decay rate. Higher = smoother. Default 0.9999 (StyleGAN2 default).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        # Shadow parameters (EMA copy)
        self.shadow: dict[str, torch.Tensor] = {}
        # Backup of original parameters (for restore)
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """Swap EMA weights into the model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()},
                "decay": self.decay}

    def load_state_dict(self, d: dict) -> None:
        device = next(self.model.parameters()).device
        self.decay = d.get("decay", self.decay)
        for k, v in d["shadow"].items():
            if k in self.shadow:
                self.shadow[k] = v.to(device)
