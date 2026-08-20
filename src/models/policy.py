"""Top-level policies: BC baseline (Week 1 Day 2) and Flow Matching policy
(Week 1 Day 4-5), plus the EMA wrapper and OT-CFM loss used to train the latter.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import LanguageEncoder, ProprioEncoder, VisualEncoder
from .denoiser import FMDenoiser


class BCPolicy(nn.Module):
    """Direct regression baseline: encoders -> concat -> MLP -> action chunk.

    TODO (w1d2t0): wire up VisualEncoder/LanguageEncoder/ProprioEncoder,
    mean-pool the visual tokens, concat all three pooled features, MLP head
    outputs a flattened (act_horizon * act_dim) chunk.
    """

    def __init__(self, act_dim: int = 7, act_horizon: int = 16, d_model: int = 256):
        super().__init__()
        self.act_dim = act_dim
        self.act_horizon = act_horizon
        raise NotImplementedError("w1d2t0")

    def forward(self, rgb, proprio, instructions):
        raise NotImplementedError


class FMPolicy(nn.Module):
    """Flow Matching policy: shared encoders feed a FMDenoiser trained with OT-CFM.

    `encode_obs` produces the `context` tensor consumed by the denoiser both at
    train time (fm_loss) and at inference time (Euler rollout).

    TODO (w1d3t1 / w1d4t0): assemble encoders + FMDenoiser here.
    """

    def __init__(self, act_dim: int = 7, act_horizon: int = 16, d_model: int = 256):
        super().__init__()
        self.act_dim = act_dim
        self.act_horizon = act_horizon
        raise NotImplementedError("w1d3t1")

    def encode_obs(self, rgb, proprio, instructions):
        raise NotImplementedError


class EMA:
    """Exponential moving average of a model's weights, used at eval/rollout time.

    TODO (w1d4t0): deepcopy the model, update() blends online weights in with
    `decay`, applied under torch.no_grad().
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema_model = deepcopy(model)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        raise NotImplementedError("w1d4t0")


def fm_loss(denoiser: nn.Module, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """OT-CFM training objective: || v_theta(x_t, t, context) - (x1 - x0) ||^2.

    x0 ~ N(0, I) noise, x1 = ground-truth action chunk, t ~ Uniform(0, 1),
    x_t = (1-t) x0 + t x1 is the straight-line interpolant.

    TODO (w1d4t0): implement per the derivation from Week 0 Day 1 (w0d1t1).
    """
    raise NotImplementedError("w1d4t0")
