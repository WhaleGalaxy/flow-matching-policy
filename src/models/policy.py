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

    与 FM 共用同一套编码器。但**默认配置下并非等参数量**：hidden=1024 时
    可训练参数 1.77M，而 FM/DDPM 是 6.80M（3.8 倍差距）。所以 FM vs BC 的
    差异里混有容量因素，不能单独归因于"确定性回归 vs 生成式流匹配"。

    要做等容量对照请用 hidden=2364（6.81M）。真正做到同骨干、同参数量的
    受控对比是 FM vs DDPM —— 两者都是 6.80M，编码器与 denoiser 结构完全相同。
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


class BCXAttnPolicy(nn.Module):
    """行为克隆基线，但**读出路径与 FM 逐层相同**。

    为什么需要这个类：原来的 `BCPolicy` 把 83 个 context token 平均池化再过 MLP。
    在接了 `goal_pos` 的配置下，目标只占 1 个 token，平均池化把它稀释成 1/83；
    而 FM 通过 cross-attention 可以定向地把注意力放到那一个 token 上。于是
    "FM 75% vs BC 4%" 这个差距里混着两件事：**建模方式**（生成 vs 回归）和
    **读出方式**（cross-attention vs 平均池化）。不拆开就不能说 FM 赢在建模上。

    这里换掉的只有建模方式：同一个 `FMDenoiser` 骨干、同样的 cross-attention、
    同样的参数量，但没有噪声输入、没有流时间、没有采样循环 —— 输入是一个学出来
    的常量 query，一次前向直接回归动作块，损失是 MSE。它与 FM 的差异因此**只剩
    "确定性回归 vs 流匹配"这一条**。

    流时间固定喂 1.0（AdaLN 需要一个 t）。多出来的参数只有 query 本身
    （act_horizon × act_dim = 112 个），相对 6.80M 可忽略。
    """

    def __init__(self, d_model=256, act_dim=7, act_horizon=16, n_layers=4, n_heads=8,
                 img_size=126, proprio_dim=9, use_proprio=True, cfg_dropout=0.0,
                 global_cond: bool = False):
        super().__init__()
        self.act_dim, self.act_horizon = act_dim, act_horizon
        self.encoder = ObsEncoder(d_model, img_size, proprio_dim, use_proprio)
        self.normalizer = ActionNormalizer(act_dim)
        # 非视觉 token（语言 + 本体/目标）额外走 AdaLN 全局条件。
        # 见 FMDenoiser 里的说明：多任务下任务身份只在这两个 token 上，
        # 光靠 cross-attention 网络会直接忽略它们。
        self.denoiser = FMDenoiser(
            d_model=d_model, d_context=d_model, n_layers=n_layers,
            n_heads=n_heads, act_dim=act_dim, act_horizon=act_horizon,
            n_global_cond=(1 + int(use_proprio)) if global_cond else 0,
        )
        # 学出来的常量 query，替代 FM 的噪声输入
        self.query = nn.Parameter(torch.zeros(1, act_horizon, act_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, rgb, proprio, instructions):
        B = rgb.shape[0]
        ctx = self.encoder(rgb, proprio, list(instructions))
        t = torch.ones(B, device=rgb.device)
        return self.denoiser(self.query.expand(B, -1, -1), t, ctx)

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
                 img_size=126, proprio_dim=9, use_proprio=True, cfg_dropout=0.2,
                 ot_coupling: bool = False, time_dist: str = "uniform",
                 time_loc: float = 0.0, time_scale: float = 1.0,
                 global_cond: bool = False):
        super().__init__()
        self.act_dim, self.act_horizon = act_dim, act_horizon
        self.cfg_dropout = cfg_dropout
        # 两个只改训练目标、不改网络结构也不改推理开销的开关。默认全关，
        # 这样"同骨干同参数量"的受控对比仍然按原样可复现；打开后是另一行结果。
        self.ot_coupling = ot_coupling
        self.time_dist, self.time_loc, self.time_scale = time_dist, time_loc, time_scale
        self.encoder = ObsEncoder(d_model, img_size, proprio_dim, use_proprio)
        self.normalizer = ActionNormalizer(act_dim)
        # 非视觉 token（语言 + 本体/目标）额外走 AdaLN 全局条件。
        # 见 FMDenoiser 里的说明：多任务下任务身份只在这两个 token 上，
        # 光靠 cross-attention 网络会直接忽略它们。
        self.denoiser = FMDenoiser(
            d_model=d_model, d_context=d_model, n_layers=n_layers,
            n_heads=n_heads, act_dim=act_dim, act_horizon=act_horizon,
            n_global_cond=(1 + int(use_proprio)) if global_cond else 0,
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
        if self.ot_coupling:
            x0 = ot_couple(x0, x1)                                # batch 内最优指派
        t = sample_flow_time(x1.shape[0], x1.device, self.time_dist,
                             self.time_loc, self.time_scale)

        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1  # 直线插值
        v_target = x1 - x0                                        # 条件速度（常数）
        v_pred = self.denoiser(xt, t, context)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def predict_action(self, rgb, proprio, instructions, n_steps=10, guidance=1.0,
                       n_average: int = 1):
        """Euler 积分采样。guidance > 1 时启用 CFG。

        CFG 把有条件和无条件两路拼成一个 batch 一次前向，比跑两遍省一半开销。

        `n_average > 1` 时独立采 n_average 个动作块再取平均，也就是用蒙特卡洛
        估计条件均值 E[a|obs] 而不是从条件分布里抽一个样本。

        为什么需要这个开关：确定性回归基线（BCXAttnPolicy）输出的正是条件均值，
        而 FM 每次推理都抽一个样本。如果这批数据的条件动作分布本来就窄，
        那么"抽样本"相对"输出均值"就是净注入噪声 —— 这正好是 FM 在本项目
        多任务上输给 BC 的候选解释之一。把 n_average 调大，如果成功率向 BC 靠拢，
        就说明差距来自采样噪声而不是学到的函数本身。观测只编码一次，
        n_average 份噪声拼成一个 batch 走完积分，代价远小于跑 n_average 遍。
        """
        B = rgb.shape[0]
        device = rgb.device
        use_cfg = guidance != 1.0
        K = max(1, int(n_average))

        ctx_cond = self.encode_obs(rgb, proprio, list(instructions))
        if K > 1:
            ctx_cond = ctx_cond.repeat_interleave(K, dim=0)
        if use_cfg:
            ctx_null = self.encode_obs(rgb, proprio, [NULL_INSTRUCTION] * B)
            if K > 1:
                ctx_null = ctx_null.repeat_interleave(K, dim=0)
            context = torch.cat([ctx_cond, ctx_null], dim=0)
        else:
            context = ctx_cond

        x = torch.randn(B * K, self.act_horizon, self.act_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B * K,), i * dt, device=device)
            if use_cfg:
                v = self.denoiser(x.repeat(2, 1, 1), t.repeat(2), context)
                v_cond, v_null = v.chunk(2, dim=0)
                v = v_null + guidance * (v_cond - v_null)
            else:
                v = self.denoiser(x, t, context)
            x = x + dt * v

        if K > 1:
            x = x.view(B, K, self.act_horizon, self.act_dim).mean(dim=1)
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


@torch.no_grad()
def ddim_sample(denoiser, alphas_cumprod: torch.Tensor, x: torch.Tensor,
                n_train_steps: int, n_steps: int, context: torch.Tensor,
                guidance: float = 1.0, clip_x0: float = 4.0) -> torch.Tensor:
    """DDIM 采样循环（确定性，η=0）。提成模块级函数是为了能脱离编码器单测 ——
    这个循环藏过一个让整个基线归零的 bug，见 DDPMPolicy.predict_action。

    x        (B, T, A) 起始噪声
    context  (B, ...) 或 CFG 时的 (2B, ...)
    """
    B, device = x.shape[0], x.device
    use_cfg = guidance != 1.0
    times = torch.linspace(n_train_steps - 1, 0, n_steps).long().to(device)

    for k, i in enumerate(times):
        t_in = torch.full((B,), i.item() / n_train_steps, device=device)
        if use_cfg:
            e = denoiser(x.repeat(2, 1, 1), t_in.repeat(2), context)
            e_cond, e_null = e.chunk(2, 0)
            eps = e_null + guidance * (e_cond - e_null)
        else:
            eps = denoiser(x, t_in, context)

        a_t = alphas_cumprod[i]
        a_prev = (alphas_cumprod[times[k + 1]] if k + 1 < len(times)
                  else torch.tensor(1.0, device=device))
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if clip_x0:
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)
        x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps
    return x


def ot_couple(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """minibatch OT 耦合：在 batch 内重排 x0，使 Σ‖x0−x1‖² 最小。

    独立耦合（默认）下同一个 x_t 会被不同样本要求朝不同方向流动，回归目标
    `x₁ − x₀` 的条件方差里有一部分纯粹是配对造成的。batch 内解一次最优指派，
    把每个真实动作块配到最近的那个噪声样本上，这部分方差就消掉了：边缘路径
    不变（两边的边缘分布都没动），但条件路径更直，因而**少步积分的误差更小**。

    这正是本项目主张的落点 —— 主图的横轴是可达控制频率，少步区的成功率就是
    结论本身，所以拉直轨迹不是锦上添花。

    B=128 时代价是一次 128×128 的匈牙利算法（~毫秒级），相对一次前向可忽略。
    """
    B = x0.shape[0]
    if B < 2:
        return x0
    from scipy.optimize import linear_sum_assignment
    # cdist 在 bf16 下会掉精度到看不出差别，这里显式升到 fp32
    a = x0.reshape(B, -1).float()
    b = x1.reshape(B, -1).float()
    cost = torch.cdist(a, b).pow_(2)
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    # linear_sum_assignment 给的是 (噪声下标, 动作下标)；我们要按动作的顺序取噪声
    perm = torch.empty(B, dtype=torch.long)
    perm[torch.as_tensor(col)] = torch.as_tensor(row)
    return x0[perm.to(x0.device)]


def sample_flow_time(B: int, device, dist: str = "uniform",
                     loc: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """采样流时间 t ∈ (0,1)。

    `uniform`      t ~ U[0,1]，Lipman 等人的原始设定。
    `logitnormal`  t = sigmoid(loc + scale·z)，z ~ N(0,1)（SD3 的做法）。
                   均匀采样把一半预算花在 t≈0 和 t≈1 附近，而那两端的速度场
                   近乎平凡（t→1 时 x_t 几乎就是 x₁）。logit-normal 把样本压向
                   中段，也就是分布真正发生形变、少步积分误差主要来源的地方。
    """
    if dist == "uniform":
        return torch.rand(B, device=device)
    if dist == "logitnormal":
        return torch.sigmoid(torch.randn(B, device=device) * scale + loc)
    raise ValueError(f"未知的时间分布 {dist!r}")


def fm_loss(denoiser: nn.Module, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """独立的 OT-CFM 损失函数（供单元测试和玩具实验使用）。"""
    x0 = torch.randn_like(action)
    t = torch.rand(action.shape[0], device=action.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * action
    return F.mse_loss(denoiser(xt, t, context), action - x0)


class DDPMPolicy(nn.Module):
    """Diffusion Policy 风格的基线：与 FMPolicy **完全相同的骨干**，
    只把训练目标换成 DDPM 的 ε-prediction。

    这样对比才干净：两者的编码器、denoiser 结构、参数量、训练预算全部一致，
    唯一的差别是"预测速度场 + Euler 积分" vs "预测噪声 + 逐步去噪"。
    任何性能/延迟差异都只能归因于这个建模选择本身。

    噪声调度用 cosine（Nichol & Dhariwal），比 linear 在少步采样时更稳。

    `clip_x0` 不是可有可无的调参项，见 predict_action 里的说明 —— 关掉它，
    这个基线会在所有配置下都退化成 2% 的常数输出。
    """

    def __init__(self, d_model=256, act_dim=7, act_horizon=16, n_layers=4, n_heads=8,
                 img_size=126, proprio_dim=9, use_proprio=True, cfg_dropout=0.2,
                 n_train_steps: int = 100, clip_x0: float = 4.0,
                 global_cond: bool = False):
        super().__init__()
        self.act_dim, self.act_horizon = act_dim, act_horizon
        self.cfg_dropout = cfg_dropout
        self.n_train_steps = n_train_steps
        self.clip_x0 = clip_x0
        self.encoder = ObsEncoder(d_model, img_size, proprio_dim, use_proprio)
        self.normalizer = ActionNormalizer(act_dim)
        # 非视觉 token（语言 + 本体/目标）额外走 AdaLN 全局条件。
        # 见 FMDenoiser 里的说明：多任务下任务身份只在这两个 token 上，
        # 光靠 cross-attention 网络会直接忽略它们。
        self.denoiser = FMDenoiser(
            d_model=d_model, d_context=d_model, n_layers=n_layers,
            n_heads=n_heads, act_dim=act_dim, act_horizon=act_horizon,
            n_global_cond=(1 + int(use_proprio)) if global_cond else 0,
        )
        self.register_buffer("alphas_cumprod", self._cosine_schedule(n_train_steps))

    @staticmethod
    def _cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        t = torch.linspace(0, T, T + 1) / T
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
        return torch.cumprod(1 - betas, dim=0)

    def encode_obs(self, rgb, proprio, instructions):
        return self.encoder(rgb, proprio, instructions)

    def compute_loss(self, batch):
        instructions = list(batch["instruction"])
        if self.training and self.cfg_dropout > 0:
            mask = torch.rand(len(instructions)) < self.cfg_dropout
            instructions = [NULL_INSTRUCTION if m else s
                            for s, m in zip(instructions, mask.tolist())]

        context = self.encode_obs(batch["rgb"], batch["proprio"], instructions)
        x0 = self.normalizer.normalize(batch["action"])           # 干净动作
        B = x0.shape[0]
        i = torch.randint(0, self.n_train_steps, (B,), device=x0.device)
        a = self.alphas_cumprod[i][:, None, None]
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise                # 前向加噪

        # denoiser 复用同一个网络，t 归一化到 [0,1] 喂给它的时间嵌入
        eps_pred = self.denoiser(xt, i.float() / self.n_train_steps, context)
        return F.mse_loss(eps_pred, noise)                        # ε-prediction

    @torch.no_grad()
    def predict_action(self, rgb, proprio, instructions, n_steps=100, guidance=1.0,
                       n_average: int = 1):
        """DDIM 采样（确定性，η=0），n_steps 可小于训练步数。

        **必须裁剪 x0_pred，否则采样在第一步就崩。** cosine 调度把 beta 上限
        clamp 在 0.999，重新 cumprod 回去之后最后一项被压成
        `alphas_cumprod[-1] ≈ 2.4e-7`，比前一项小 1000 倍（其余相邻步的比值都在
        1–2 之间）。而采样恰好从这一项起步，于是

            x0_pred = (x − √(1−ᾱ)·ε) / √ᾱ        1/√ᾱ ≈ 2029

        把 ε 预测的误差放大两千倍。下一步再乘回 √ᾱ_prev 只抵消一部分，净放大
        约 31.6 倍，采样值直接飞掉，动作被环境裁到边界，策略退化成常数输出。

        症状：**所有配置下成功率完全相同**（本项目实测采样步数 1→50、执行长度
        1→16 全是 2.0%，平均步长恒为 294.1）。这个"各配置精确相等"的指纹在
        docs/debugging.md 里出现过两次，都不是建模问题。

        裁剪是 diffusers (`clip_sample=True`) 和 Diffusion Policy 的标准做法。
        动作已归一化到单位方差，所以 ±4 是 4σ，不会切掉真实动作。
        """
        B, device = rgb.shape[0], rgb.device
        use_cfg = guidance != 1.0
        K = max(1, int(n_average))   # 见 FMPolicy.predict_action 里的说明

        ctx = self.encode_obs(rgb, proprio, list(instructions))
        if K > 1:
            ctx = ctx.repeat_interleave(K, dim=0)
        if use_cfg:
            null = self.encode_obs(rgb, proprio, [NULL_INSTRUCTION] * B)
            if K > 1:
                null = null.repeat_interleave(K, dim=0)
            ctx = torch.cat([ctx, null], 0)

        x = torch.randn(B * K, self.act_horizon, self.act_dim, device=device)
        x = ddim_sample(self.denoiser, self.alphas_cumprod, x, self.n_train_steps,
                        n_steps, ctx, guidance=guidance, clip_x0=self.clip_x0)
        if K > 1:
            x = x.view(B, K, self.act_horizon, self.act_dim).mean(dim=1)
        return self.normalizer.denormalize(x)


def trainable_state_dict(model: nn.Module, reference: nn.Module | None = None) -> dict:
    """只保留可训练参数和全部 buffer，丢掉冻结的骨干权重。

    完整 state_dict 里 99% 是冻结的 DINOv2 + SigLIP 权重（单个 checkpoint 1.1GB），
    它们完全由 timm/HF 的预训练标识决定，重新加载即可，没必要随每个 checkpoint 存一份。
    buffer 必须保留 —— 归一化统计量和噪声调度都在里面。

    `reference` 指定用谁来判定"哪些参数是可训练的"。默认用 `model` 自己，但保存
    **EMA 副本**时必须显式传入在线模型：`EMA.__init__` 会把副本所有参数的
    requires_grad 置为 False，拿副本自己判定会得到空集，于是全部权重被误当成
    冻结骨干丢掉，存出一个只剩 buffer 的空 checkpoint。
    """
    ref = model if reference is None else reference
    trainable = {n for n, p in ref.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    assert trainable, (
        "没有任何可训练参数被选中 —— 若在保存 EMA 副本，需传入 reference=在线模型")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in trainable or k in buffers}


def load_trainable_state_dict(model: nn.Module, state: dict) -> None:
    """载入 trainable_state_dict 保存的权重；冻结骨干保持预训练初始化。"""
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"checkpoint 里有模型不认识的键: {unexpected[:5]}"
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    unexplained = [k for k in missing if k not in frozen]
    assert not unexplained, f"缺失的键不属于冻结骨干: {unexplained[:5]}"
