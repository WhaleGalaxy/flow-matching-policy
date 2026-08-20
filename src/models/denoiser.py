"""Flow Matching denoiser: Transformer with AdaLN(t) + cross-attention to context.

Building blocks land in Week 0 Day 2 (attention primitives, checklist ids
w0d2t0/t1/t2) and get assembled into the full denoiser in Week 1 Day 3
(w1d3t0/t1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of a scalar timestep t in [0, 1].

    TODO (w1d3t0): standard transformer sinusoidal embedding, t scaled by 1000
    before taking sin/cos so t in [0,1] spans a useful frequency range.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("w1d3t0")


class AdaLN(nn.Module):
    """Adaptive LayerNorm: condition a token sequence on a pooled embedding (here, t).

    TODO (w0d2t2): parameter-free LayerNorm, then per-sample scale/shift
    predicted from t_emb via two linear layers (DiT-style).
    """

    def __init__(self, d_model: int = 256, d_cond: int = 256):
        super().__init__()
        raise NotImplementedError("w0d2t2")

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CrossAttention(nn.Module):
    """Action tokens attend to context tokens (obs + language + proprio).

    TODO (w0d2t1): multi-head cross-attention, Q from x, K/V from context.
    Sanity check first with a from-scratch scaled dot-product attention
    (w0d2t0) before wiring up the multi-head module here.
    """

    def __init__(self, d_model: int = 256, d_context: int = 256, n_heads: int = 8):
        super().__init__()
        raise NotImplementedError("w0d2t1")

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FMDenoiser(nn.Module):
    """Predicts the velocity field v_theta(x_t, t, context).

    Input:
      noisy_action (B, H, act_dim)  -- x_t along the flow
      t            (B,)             -- flow time in [0, 1]
      context      (B, L, d_model)  -- concatenated obs/lang/proprio tokens
    Output:
      velocity (B, H, act_dim)

    TODO (w1d3t1): per-layer AdaLN -> self-attention -> cross-attention -> FFN,
    stack n_layers of these, output-project back to act_dim.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        act_dim: int = 7,
        act_horizon: int = 16,
    ):
        super().__init__()
        raise NotImplementedError("w1d3t1")

    def forward(
        self, noisy_action: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
