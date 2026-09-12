"""在 latent 动作空间里做流匹配，值不值得试？—— 训练前几分钟的判据。

动机来自两个实测数字（见 docs/debugging.md 第六轮）：

  * FM 用 16 个样本估计的条件均值，开环 RMSE 是 0.229；等条件 BC 是 0.154。
    同等容量下建模整个条件分布，代价是条件均值本身估得更差。
  * 对 FM 的样本取平均，开环误差从 0.284 降到 0.229，**闭环成功率却从 60% 掉到 53%**。
    也就是说，**动作块的集合不是凸的** —— 两个合法动作块的平均不是合法动作块
    （夹爪近乎二值，不同样本里闭合时刻不同，平均后是一段悬在中间的值）。

于是一个自然的想法是：把流匹配搬到学出来的 latent 动作空间里。降维、让平均和插值
落回合法动作块、把时序相关性交给解码器。

但这条路有一个会一票否决的风险：**自编码器的重建误差是整条路线的误差下界**。
如果重建 RMSE 已经接近甚至超过 BC 的 0.154，那再好的流模型也赚不回来。

这个脚本就报三件事，全部在归一化动作单位下（数据的边缘标准差按定义为 1）：

  1. 各 latent 维度下的重建 RMSE，以及它与 BC 的 0.154 的关系 —— **头顶还剩多少空间**
  2. 原始空间里平均两个动作块 vs latent 空间里平均再解码，哪个更像真实动作
     （量夹爪的"中间值"占比：合法动作块的夹爪几乎总在 ±1 附近）
  3. latent 空间里的条件分布好不好建（各维方差是否均衡、是否近高斯）

    python scripts/latent_action_probe.py --task PickCube-v1 --dims 8 16 32 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ManiskillDataset, default_h5_path  # noqa: E402


class ChunkAE(nn.Module):
    """动作块自编码器。故意做小：它是"预处理"，不该把容量预算吃掉。"""

    def __init__(self, horizon=16, act_dim=7, d_latent=16, hidden=512):
        super().__init__()
        n = horizon * act_dim
        self.horizon, self.act_dim = horizon, act_dim
        self.enc = nn.Sequential(nn.Linear(n, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n))

    def encode(self, x):
        return self.enc(x.flatten(1))

    def decode(self, z):
        return self.dec(z).view(-1, self.horizon, self.act_dim)

    def forward(self, x):
        return self.decode(self.encode(x))


def gripper_midband(chunks: torch.Tensor, lo=-0.5, hi=0.5) -> float:
    """夹爪落在"中间值"的比例。

    演示里夹爪几乎总在 ±1 附近（张开/闭合），停在中间是物理上没有意义的指令。
    这个比例就是"这批动作块像不像真的"的一个直接判据。
    """
    g = chunks[..., -1]
    return float(((g > lo) & (g < hi)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--sources", nargs="+", default=["motionplanning"])
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--act-horizon", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=60000)
    ap.add_argument("--bc-error", type=float, default=0.154,
                    help="等条件 BC 的开环动作 RMSE，作为『头顶还剩多少空间』的参照")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ---- 取动作块（只读动作，不读图像，秒级）----
    chunks = []
    for src in args.sources:
        ds = ManiskillDataset(default_h5_path(args.task, source=src), task=args.task,
                              act_horizon=args.act_horizon, source=src)
        f = ds.file
        for name, T in ds.episode_lengths.items():
            a = np.asarray(f[name]["actions"], dtype=np.float32)
            idx = np.minimum(np.arange(T)[:, None] + np.arange(args.act_horizon)[None, :],
                             T - 1)
            chunks.append(a[idx])
    X = torch.from_numpy(np.concatenate(chunks))
    rng = np.random.default_rng(0)
    if len(X) > args.n_samples:
        X = X[rng.choice(len(X), args.n_samples, replace=False)]

    mean, std = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0).clamp_min(1e-6)
    Xn = ((X - mean) / std).to(args.device)
    n_val = len(Xn) // 10
    Xtr, Xva = Xn[n_val:], Xn[:n_val]

    print(f"\n===== latent 动作空间判据：{args.task} / {'+'.join(args.sources)} =====")
    print(f"{len(Xn):,} 个动作块（{args.act_horizon}×{X.shape[-1]}），"
          f"留出 {len(Xva):,} 个做验证")
    print(f"归一化动作单位，边缘标准差按定义为 1.000；"
          f"等条件 BC 的开环 RMSE = {args.bc_error:.3f}\n")
    print(f"  {'latent 维度':<14}{'重建 RMSE':>12}{'占 BC 误差':>12}{'头顶空间':>12}")

    best = None
    for d in args.dims:
        torch.manual_seed(0)
        ae = ChunkAE(args.act_horizon, X.shape[-1], d).to(args.device)
        opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(args.steps):
            b = Xtr[torch.randint(0, len(Xtr), (args.batch_size,), device=args.device)]
            loss = nn.functional.mse_loss(ae(b), b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            rec = ae(Xva)
            rmse = float((rec - Xva).pow(2).mean(dim=(1, 2)).sqrt().median())
        ratio = rmse / args.bc_error
        verdict = "没有空间" if ratio >= 0.9 else ("偏紧" if ratio >= 0.5 else "有空间")
        print(f"  {d:<14}{rmse:>12.3f}{ratio:>11.0%}{verdict:>13}")
        if best is None or rmse < best[1]:
            best = (d, rmse, ae)

    d, rmse, ae = best
    print(f"\n  最好的是 latent={d}，重建 RMSE {rmse:.3f}")

    # ---- 凸性：原始空间平均 vs latent 空间平均 ----
    with torch.no_grad():
        i = torch.randperm(len(Xva), device=args.device)[:4000]
        j = torch.randperm(len(Xva), device=args.device)[:4000]
        a, b = Xva[i], Xva[j]
        raw_avg = (a + b) / 2
        lat_avg = ae.decode((ae.encode(a) + ae.encode(b)) / 2)
    # 换回原始动作单位再看夹爪，因为 ±1 是原始单位里的物理量
    to_raw = lambda t: t.cpu() * std + mean
    print(f"\n  夹爪停在中间值（|g|<0.5）的比例 —— 合法动作块里它应该很小：")
    print(f"    真实演示的动作块        {gripper_midband(to_raw(Xva)):.1%}")
    print(f"    原始空间里平均两个块    {gripper_midband(to_raw(raw_avg)):.1%}")
    print(f"    latent 空间里平均再解码 {gripper_midband(to_raw(lat_avg)):.1%}")
    print("    第二行远大于第一行 = 动作块的集合不是凸的，"
          "这正是对 FM 的样本取平均会掉成功率的原因；")
    print("    第三行若明显小于第二行，说明 latent 空间把这件事修好了。")

    # ---- latent 分布好不好建 ----
    with torch.no_grad():
        Z = ae.encode(Xva)
    zs = Z.std(0)
    print(f"\n  latent 各维标准差：中位数 {float(zs.median()):.3f}  "
          f"最大/最小 {float(zs.max()/zs.clamp_min(1e-6).min()):.1f}×")
    print("    比值越接近 1，各维尺度越均衡，流模型越好建；"
          "相差几十倍时应加 KL 或做逐维白化。")


if __name__ == "__main__":
    main()
