"""顶层策略：BC 基线 与 Flow Matching 策略，含 EMA、CFG 与 Euler 采样。"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoiser import FMDenoiser
from .encoders import LanguageEncoder, ProprioEncoder, VisualEncoder

NULL_INSTRUCTION = ""  # CFG 的 null condition


class ObsEncoder(nn.Module):
    """把三路观测编码并拼成一个 context 序列。BC 和 FM 共用，保证对比公平。"""

    def __init__(self, d_model=256, img_size=126, proprio_dim=9, use_proprio=True):
        super().__init__()
        self.visual = VisualEncoder(d_model=d_model, img_size=img_size)
        self.language = LanguageEncoder(d_model=d_model)
        self.use_proprio = use_proprio
        if use_proprio:
            self.proprio = ProprioEncoder(proprio_dim=proprio_dim, d_model=d_model)
        self.context_len = self.visual.n_patches + 1 + int(use_proprio)

    def forward(self, rgb, proprio, instructions) -> torch.Tensor:
        tokens = [self.visual(rgb), self.language(instructions)]
        if self.use_proprio:
            tokens.append(self.proprio(proprio))
        return torch.cat([t.to(tokens[0].dtype) for t in tokens], dim=1)


class ActionNormalizer(nn.Module):
    """动作归一化。统计量存成 buffer，随 checkpoint 一起保存，
    避免推理时忘了带 stats —— 这是这类项目最常见的 bug 之一。"""

    def __init__(self, act_dim: int = 7):
        super().__init__()
        self.register_buffer("mean", torch.zeros(act_dim))
        self.register_buffer("std", torch.ones(act_dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, actions: torch.Tensor) -> None:
        flat = actions.reshape(-1, actions.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-6))
        self.fitted.fill_(True)

    def normalize(self, a):
        return (a - self.mean) / self.std

    def denormalize(self, a):
        return a * self.std + self.mean


class BCPolicy(nn.Module):
    """行为克隆基线：context 池化后过 MLP，直接回归整个动作块。

    刻意与 FM 共用同一套编码器和相近的参数量，这样 FM vs BC 的差异
    只来自建模方式（确定性回归 vs 生成式流匹配），而不是模型容量。
    """

    def __init__(self, d_model=256, act_dim=7, act_horizon=16,
                 img_size=126, proprio_dim=9, use_proprio=True, hidden=1024):
        super().__init__()
        self.act_dim, self.act_horizon = act_dim, act_horizon
        self.encoder = ObsEncoder(d_model, img_size, proprio_dim, use_proprio)
        self.normalizer = ActionNormalizer(act_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_horizon * act_dim),
        )

    def forward(self, rgb, proprio, instructions):
        ctx = self.encoder(rgb, proprio, instructions).mean(dim=1)  # 平均池化
        return self.head(ctx).view(-1, self.act_horizon, self.act_dim)

    def compute_loss(self, batch):
        target = self.normalizer.normalize(batch["action"])
        pred = self(batch["rgb"], batch["proprio"], batch["instruction"])
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict_action(self, rgb, proprio, instructions, **_):
        return self.normalizer.denormalize(self(rgb, proprio, instructions))


class FMPolicy(nn.Module):
    """Flow Matching 策略：编码器产生 context，denoiser 回归速度场。"""

    def __init__(self, d_model=256, act_dim=7, act_horizon=16, n_layers=4, n_heads=8,
                 img_size=126, proprio_dim=9, use_proprio=True, cfg_dropout=0.2):
        super().__init__()
        self.act_dim, self.act_horizon = act_dim, act_horizon
        self.cfg_dropout = cfg_dropout
        self.encoder = ObsEncoder(d_model, img_size, proprio_dim, use_proprio)
        self.normalizer = ActionNormalizer(act_dim)
        self.denoiser = FMDenoiser(
            d_model=d_model, d_context=d_model, n_layers=n_layers,
            n_heads=n_heads, act_dim=act_dim, act_horizon=act_horizon,
        )

    def encode_obs(self, rgb, proprio, instructions):
        return self.encoder(rgb, proprio, instructions)

    def compute_loss(self, batch):
        """OT-CFM 目标： E ‖ v_θ((1-t)x₀ + t·x₁, t, ctx) − (x₁ − x₀) ‖²"""
        instructions = list(batch["instruction"])
        # CFG：训练时按概率把指令替换成 null，让同一个网络兼学有/无条件两种速度场
        if self.training and self.cfg_dropout > 0:
            mask = torch.rand(len(instructions)) < self.cfg_dropout
            instructions = [NULL_INSTRUCTION if m else s
                            for s, m in zip(instructions, mask.tolist())]

        context = self.encode_obs(batch["rgb"], batch["proprio"], instructions)
        x1 = self.normalizer.normalize(batch["action"])          # 真实动作
        x0 = torch.randn_like(x1)                                 # 噪声
        t = torch.rand(x1.shape[0], device=x1.device)             # t ~ U[0,1]

        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1  # 直线插值
        v_target = x1 - x0                                        # 条件速度（常数）
        v_pred = self.denoiser(xt, t, context)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def predict_action(self, rgb, proprio, instructions, n_steps=10, guidance=1.0):
        """Euler 积分采样。guidance > 1 时启用 CFG。

        CFG 把有条件和无条件两路拼成一个 batch 一次前向，比跑两遍省一半开销。
        """
        B = rgb.shape[0]
        device = rgb.device
        use_cfg = guidance != 1.0

        ctx_cond = self.encode_obs(rgb, proprio, list(instructions))
        if use_cfg:
            ctx_null = self.encode_obs(rgb, proprio, [NULL_INSTRUCTION] * B)
            context = torch.cat([ctx_cond, ctx_null], dim=0)
        else:
            context = ctx_cond

        x = torch.randn(B, self.act_horizon, self.act_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            if use_cfg:
                v = self.denoiser(x.repeat(2, 1, 1), t.repeat(2), context)
                v_cond, v_null = v.chunk(2, dim=0)
                v = v_null + guidance * (v_cond - v_null)
            else:
                v = self.denoiser(x, t, context)
            x = x + dt * v

        return self.normalizer.denormalize(x)


class EMA:
    """权重的指数滑动平均。推理一律用 EMA 权重 —— 生成模型上这通常
    带来数个百分点的稳定提升，忘了用是另一个常见 bug。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: int = 1000):
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.step = decay, warmup, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        # 预热期用较小的 decay，否则 EMA 会长时间停留在随机初始化附近
        d = min(self.decay, (1 + self.step) / (10 + self.step)) if self.step < self.warmup else self.decay
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.lerp_(p.detach(), 1 - d)
        for eb, b in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def fm_loss(denoiser: nn.Module, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """独立的 OT-CFM 损失函数（供单元测试和玩具实验使用）。"""
    x0 = torch.randn_like(action)
    t = torch.rand(action.shape[0], device=action.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * action
    return F.mse_loss(denoiser(xt, t, context), action - x0)
