"""Week0 Day3：2D Swiss Roll 上的完整 Flow Matching。

训练循环只有四行（采 t / 采噪声 / 插值 / 回归 x1-x0），但足以复现 FM 的全部核心行为。
除了生成质量，还专门画出**采样轨迹**，用来验证笔记里那个关键论断：
条件路径是直线，但边缘轨迹是弯的。
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_swiss_roll

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


def swiss_roll(n):
    X, _ = make_swiss_roll(n, noise=0.5)
    X = X[:, [0, 2]] / 5.0                      # 取 2D 切面并缩放到 ~N(0,1) 量级
    return torch.tensor(X, dtype=torch.float32)


class VelocityField(nn.Module):
    """v_θ(x, t)：把 t 直接 concat 进输入，2D 玩具问题上足够了。"""

    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def train(data, steps=8000, bs=512, lr=3e-4):
    model = VelocityField().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(DEV)

    for step in range(steps):
        x1 = data[torch.randint(len(data), (bs,), device=DEV)]   # 真实样本
        x0 = torch.randn_like(x1)                                # 噪声
        t = torch.rand(bs, 1, device=DEV)                        # t ~ U[0,1]

        xt = (1 - t) * x0 + t * x1                               # 直线插值
        v_target = x1 - x0                                       # 条件速度（常数！）

        loss = (model(xt, t) - v_target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 2000 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def sample(model, n, n_steps, return_traj=False):
    """Euler 积分：x_{t+Δ} = x_t + Δ·v_θ(x_t, t)，从 t=0 积到 t=1。"""
    x = torch.randn(n, 2, device=DEV)
    traj = [x.clone()]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n, 1), i * dt, device=DEV)
        x = x + dt * model(x, t)
        traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


def sliced_w2(a, b, n_proj=256):
    """切片 2-Wasserstein 距离：往随机方向投影，比较 1D 分布，取平均。"""
    d = a.shape[1]
    dirs = torch.randn(d, n_proj, device=a.device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    pa = (a @ dirs).sort(dim=0).values
    pb = (b @ dirs).sort(dim=0).values
    return (pa - pb).pow(2).mean().sqrt().item()


def main():
    print(f"device: {DEV}")
    data = swiss_roll(20000)
    holdout = swiss_roll(4000).to(DEV)

    print("训练 velocity field ...")
    model = train(data)

    step_counts = [1, 2, 5, 10, 50, 100]
    print("\n采样步数 vs 生成质量（切片 W2，越小越好）：")
    results = {}
    for k in step_counts:
        s = sample(model, 4000, k)
        results[k] = (s, sliced_w2(s, holdout))
        print(f"  {k:3d} 步   sliced-W2 = {results[k][1]:.4f}")

    # ---- 作图 ----
    fig = plt.figure(figsize=(16, 7))
    lim = 3.0

    ax = fig.add_subplot(2, 4, 1)
    d = data[:4000].numpy()
    ax.scatter(d[:, 0], d[:, 1], s=1, alpha=0.4, c="#333")
    ax.set_title("Ground truth", fontsize=11)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])

    for i, k in enumerate(step_counts):
        ax = fig.add_subplot(2, 4, i + 2)
        s = results[k][0].cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=1, alpha=0.4, c="#378ADD")
        ax.set_title(f"{k} Euler step{'s' if k > 1 else ''}\nW2={results[k][1]:.3f}", fontsize=11)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])

    # 轨迹图：证明边缘轨迹是弯的
    ax = fig.add_subplot(2, 4, 8)
    _, traj = sample(model, 150, 100, return_traj=True)
    traj = traj.cpu().numpy()
    for j in range(traj.shape[1]):
        ax.plot(traj[:, j, 0], traj[:, j, 1], lw=0.5, alpha=0.5, c="#C4682B")
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=4, c="#888", label="t=0 (noise)")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=4, c="#0C447C", label="t=1 (data)")
    ax.set_title("Marginal trajectories\n(curved, not straight!)", fontsize=11)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out = "/home/xjy/flow_matching/fm_policy/notebooks/w0d3_swiss_roll_fm.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
