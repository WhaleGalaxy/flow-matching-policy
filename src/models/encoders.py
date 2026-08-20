"""Observation encoders: vision (DINOv2), language (SigLIP), proprioception.

Filled in during Week 1, Day 1-2 (checklist ids w1d0t0, w1d1t0, w1d1t1).
Each encoder maps its modality to a sequence of `d_model`-dim tokens so they
can all be concatenated into one `context` tensor for the FM denoiser's
cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Frozen DINOv2-Small -> per-patch tokens, projected to d_model.

    Input:  imgs (B, T, 3, H, W)  -- T stacked frames (obs_horizon)
    Output: tokens (B, N_patches, d_model)

    TODO (w1d0t0): load `vit_small_patch14_dinov2.lvd142m` via timm,
    freeze it, average pool over the time dimension, project 384 -> d_model.
    TODO (w1d0t1): verify freezing actually stops gradients
    (sum(p.requires_grad for p in self.dino.parameters()) == 0).
    """

    def __init__(self, d_model: int = 256, freeze: bool = True):
        super().__init__()
        self.d_model = d_model
        self.freeze = freeze
        raise NotImplementedError("w1d0t0: build DINOv2 backbone + projection head")

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LanguageEncoder(nn.Module):
    """Frozen SigLIP text tower -> single language token, with a per-instruction cache.

    Input:  instructions (list[str]) of length B
    Output: tokens (B, d_model)

    TODO (w1d1t0): load `google/siglip-base-patch16-224` text model, freeze it,
    project 768 -> d_model, cache embeddings by instruction string.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        raise NotImplementedError("w1d1t0: build SigLIP text encoder + projection + cache")

    def forward(self, instructions: list[str]) -> torch.Tensor:
        raise NotImplementedError


class ProprioEncoder(nn.Module):
    """Small MLP over the latest proprioceptive state (joint angles + end-effector pose).

    Input:  proprio (B, T, proprio_dim)
    Output: tokens (B, 1, d_model)

    TODO (w1d1t1): 2-layer MLP, use only the most recent timestep.
    """

    def __init__(self, proprio_dim: int = 9, d_model: int = 256):
        super().__init__()
        raise NotImplementedError("w1d1t1: build proprio MLP")

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
