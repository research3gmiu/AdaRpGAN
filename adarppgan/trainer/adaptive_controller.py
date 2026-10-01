"""
AdaptiveController
==================
The core contribution of AdaRpGAN.

Jointly adapts:
  γ  (gradient-penalty weight) via an EMA of Lipschitz violation
  n_critic (D steps per G step) via a rolling-window loss variance

Both signals share a common measurement: the discriminator gradient norm
returned by compute_gradient_penalty / compute_r1_penalty.

Algorithm
---------
At every D step t:

  1. Observe gp_grad_norm  → Lipschitz violation proxy  v_t = max(0, norm_t - 1)
  2. Update EMA of v:       v̄_t = β·v̄_{t-1} + (1-β)·v_t
  3. Adjust γ:              γ_t = γ_{t-1} · exp(α_γ · (v̄_t - v_target))
                            clip to [γ_min, γ_max]

  4. Observe D loss → push into rolling buffer of length W
  5. Compute σ²(D_loss over last W steps)
  6. Adjust n_critic:
       if σ² > σ_high  and  cooldown=0  → n_critic = min(n_critic+1, n_max)
       if σ² < σ_low   and  cooldown=0  → n_critic = max(n_critic-1, 1)
       reset cooldown to C after any change

Hyperparameters with recommended defaults:
  α_γ       = 0.01   (γ step size)
  β         = 0.99   (EMA momentum)
  v_target  = 0.01   (desired mean Lipschitz violation)
  γ_min     = 1.0
  γ_max     = 50.0
  σ_high    = 0.50   (variance threshold to increase n_critic)
  σ_low     = 0.05   (variance threshold to decrease n_critic)
  W         = 50     (rolling window width)
  C         = 100    (cooldown steps after n_critic change)
  n_min     = 1
  n_max     = 5
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ControllerConfig:
    # Gamma adaptation
    gamma_init:   float = 10.0
    gamma_min:    float = 1.0
    gamma_max:    float = 50.0
    alpha_gamma:  float = 0.01
    ema_beta:     float = 0.99
    viol_target:  float = 0.01

    # n_critic adaptation
    n_critic_init: int   = 1
    n_critic_min:  int   = 1
    n_critic_max:  int   = 5
    window:        int   = 50
    sigma_high:    float = 0.50
    sigma_low:     float = 0.05
    cooldown:      int   = 100


@dataclass
class ControllerState:
    """Serialisable snapshot of controller state (for checkpointing)."""
    gamma:         float
    n_critic:      int
    viol_ema:      float
    loss_buffer:   list
    cooldown_left: int
    step:          int


class AdaptiveController:
    """
    Stateful adaptive hyperparameter controller for AdaRpGAN.

    Usage
    -----
    ctrl = AdaptiveController(ControllerConfig())

    # Inside D training loop:
    ctrl.observe(grad_norm=..., d_loss=...)

    # Read current hyperparams:
    gamma    = ctrl.gamma
    n_critic = ctrl.n_critic

    # Log for diagnostics:
    info = ctrl.info()
    """

    def __init__(self, cfg: ControllerConfig = ControllerConfig()):
        self.cfg = cfg

        self.gamma:      float = cfg.gamma_init
        self.n_critic:   int   = cfg.n_critic_init

        self._viol_ema:     float        = 0.0
        self._loss_buf:     deque        = deque(maxlen=cfg.window)
        self._cooldown_left: int         = 0
        self._step:         int          = 0

        # History for logging / plotting
        self.history_gamma:    list[float] = []
        self.history_n_critic: list[int]   = []
        self.history_viol:     list[float] = []
        self.history_var:      list[float] = []

    # ──────────────────────────────────────────
    # Main update
    # ──────────────────────────────────────────

    def observe(
        self,
        grad_norm: float,
        d_loss:    float,
    ) -> None:
        """
        Call once per D optimiser step with:
          grad_norm : mean ||∇D(x̂)||₂ for this step  (scalar float)
          d_loss    : D total loss value              (scalar float)
        """
        cfg = self.cfg
        self._step += 1

        # ── 1. Lipschitz violation ───────────────
        viol = max(0.0, grad_norm - 1.0)
        self._viol_ema = cfg.ema_beta * self._viol_ema + (1.0 - cfg.ema_beta) * viol

        # ── 2. Gamma update ──────────────────────
        log_adjust = cfg.alpha_gamma * (self._viol_ema - cfg.viol_target)
        self.gamma = self.gamma * math.exp(log_adjust)
        self.gamma = max(cfg.gamma_min, min(cfg.gamma_max, self.gamma))

        # ── 3. Loss variance ────────────────────
        self._loss_buf.append(d_loss)
        loss_var = _buffer_variance(self._loss_buf)

        # ── 4. n_critic update (with cooldown) ──
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        else:
            if loss_var > cfg.sigma_high and self.n_critic < cfg.n_critic_max:
                self.n_critic      += 1
                self._cooldown_left = cfg.cooldown
            elif loss_var < cfg.sigma_low and self.n_critic > cfg.n_critic_min:
                self.n_critic      -= 1
                self._cooldown_left = cfg.cooldown

        # ── 5. History ───────────────────────────
        self.history_gamma.append(self.gamma)
        self.history_n_critic.append(self.n_critic)
        self.history_viol.append(self._viol_ema)
        self.history_var.append(loss_var)

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────

    def info(self) -> dict:
        return {
            "step":      self._step,
            "gamma":     round(self.gamma, 4),
            "n_critic":  self.n_critic,
            "viol_ema":  round(self._viol_ema, 5),
            "loss_var":  round(_buffer_variance(self._loss_buf), 5),
            "cooldown":  self._cooldown_left,
        }

    # ──────────────────────────────────────────
    # Checkpoint support
    # ──────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "gamma":         self.gamma,
            "n_critic":      self.n_critic,
            "viol_ema":      self._viol_ema,
            "loss_buffer":   list(self._loss_buf),
            "cooldown_left": self._cooldown_left,
            "step":          self._step,
        }

    def load_state_dict(self, d: dict) -> None:
        self.gamma          = d["gamma"]
        self.n_critic       = d["n_critic"]
        self._viol_ema      = d["viol_ema"]
        self._loss_buf      = deque(d["loss_buffer"], maxlen=self.cfg.window)
        self._cooldown_left = d["cooldown_left"]
        self._step          = d["step"]


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _buffer_variance(buf: deque) -> float:
    if len(buf) < 2:
        return 0.0
    n   = len(buf)
    mu  = sum(buf) / n
    var = sum((x - mu) ** 2 for x in buf) / (n - 1)
    return var
