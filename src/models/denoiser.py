"""Flow Matching denoiser：预测速度场 v_θ(x_t, t, context)。

结构上是一个 Transformer，动作块的 H 个时间步当作 token 序列：
  - self-attention  让动作块内部各步互相协调（保证轨迹连贯）
  - cross-attention 让动作 token 去查询 context（视觉 patch + 语言 + 本体）
  - AdaLN          把流时间 t 注入每一层

为什么 t 走 AdaLN 而不是当成一个 token 拼进序列：t 是全局标量条件，
对每个 token 的影响是一致的缩放/平移，用 AdaLN 表达更直接、参数也更少（DiT 的做法）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """把标量流时间 t ∈ [0,1] 编成 dim 维正弦嵌入。

    乘 1000 是因为原始的正弦编码是为整数位置设计的（波长 2π~10000·2π）。
    t 只在 [0,1] 里取值，不放大的话所有频率分量几乎不变化，等于没编码。
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, "维度必须是偶数（sin/cos 各占一半）"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class AdaLN(nn.Module):
    """Adaptive LayerNorm：用条件嵌入预测每个样本的 scale / shift。

    初始化成恒等映射（scale 和 shift 的输出层权重置零 → γ=1, β=0）。
    这是 DiT 的关键技巧：训练初期每个残差块近似恒等，深层网络也能稳定起步。
    """

    def __init__(self, d_model: int = 256, d_cond: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class CrossAttention(nn.Module):
    """动作 token 作 Q，context token 作 K/V。"""

    def __init__(self, d_model: int = 256, d_context: int = 256, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_context, d_model)
        self.to_v = nn.Linear(d_context, d_model)
        self.to_out = nn.Linear(d_model, d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,L,d)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, H, _ = x.shape
        q, k, v = self._split(self.to_q(x)), self._split(self.to_k(context)), self._split(self.to_v(context))
        out = F.scaled_dot_product_attention(q, k, v)          # 走 PyTorch 的 flash 实现
        out = out.transpose(1, 2).reshape(B, H, -1)
        return self.to_out(out)


class DenoiserBlock(nn.Module):
    """一层：AdaLN-self-attn → AdaLN-cross-attn → AdaLN-FFN，全部残差连接。

    三个子层各自带一个 AdaLN（pre-norm），且每个残差分支的输出投影初始化为零，
    使整个 block 在 t=0 时刻起步即恒等。
    """

    def __init__(self, d_model: int, d_context: int, n_heads: int):
        super().__init__()
        self.norm_sa = AdaLN(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm_ca = AdaLN(d_model, d_model)
        self.cross_attn = CrossAttention(d_model, d_context, n_heads)

        self.norm_ff = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        # 残差分支归零起步，配合 AdaLN 的零初始化让 block ≈ 恒等
        nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)
        nn.init.zeros_(self.cross_attn.to_out.weight); nn.init.zeros_(self.cross_attn.to_out.bias)

    def forward(self, x, t_emb, context):
        h = self.norm_sa(x, t_emb)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.norm_ca(x, t_emb), context)
        x = x + self.ffn(self.norm_ff(x, t_emb))
        return x


class FMDenoiser(nn.Module):
    """预测速度场 v_θ(x_t, t, context)。

    输入
      noisy_action (B, H, act_dim)  流上的 x_t
      t            (B,)             流时间 ∈ [0,1]
      context      (B, L, d_ctx)    视觉/语言/本体拼接后的条件 token
    输出
      velocity     (B, H, act_dim)
    """

    def __init__(
        self,
        d_model: int = 256,
        d_context: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        act_dim: int = 7,
        act_horizon: int = 16,
        n_global_cond: int = 0,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.act_horizon = act_horizon
        # 把 context 末尾的 n_global_cond 个 token（语言 + 本体/目标）额外作为
        # **全局条件**经 AdaLN 注入，而不只是躺在 cross-attention 的 context 里。
        #
        # 为什么需要：多任务下任务身份全在这两个 token 上，而它们是 83 个 token
        # 里的 2 个 —— 注意力可以直接不看它们，实测三任务联合训练的结果是
        # PickCube 3% / PushCube 82% / StackCube 4%，也就是策略学会了最容易的
        # 那个任务然后到处执行它。AdaLN 调制每一层的 scale/shift，绕不过去。
        # 这是 DiT 注入类别标签的做法，条件化的位置比条件化的内容更重要。
        #
        # 0 = 关闭，与既有 checkpoint 逐位兼容。
        self.n_global_cond = n_global_cond

        self.t_emb = SinusoidalPosEmb(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        )
        self.action_in = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, act_horizon, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [DenoiserBlock(d_model, d_context, n_heads) for _ in range(n_layers)]
        )
        if n_global_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(d_context, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            # 末层归零：训练起步时全局条件的贡献为 0，等价于关闭这条通路，
            # 于是它只能靠梯度**挣**到影响力，不会一上来就扰乱已经稳定的初始化。
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

        self.norm_out = AdaLN(d_model, d_model)
        self.action_out = nn.Linear(d_model, act_dim)
        # 输出层归零：训练第一步预测速度为 0，避免初期发散
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_action, t, context):
        assert noisy_action.shape[1] == self.act_horizon, (
            f"动作块长度应为 {self.act_horizon}, got {noisy_action.shape[1]}")
        t_emb = self.t_mlp(self.t_emb(t))          # (B, d_model)
        if self.n_global_cond:
            # 从 context 自己末尾取，CFG 时 context 已经是 (2B, ...)，
            # 有条件/无条件两路各自拿到自己的语言 token，不需要额外传参
            t_emb = t_emb + self.cond_mlp(context[:, -self.n_global_cond:, :].mean(1))
        x = self.action_in(noisy_action) + self.pos_emb
        for block in self.blocks:
            x = block(x, t_emb, context)
        return self.action_out(self.norm_out(x, t_emb))
