"""端到端验证 + 项目的核心科学论点。

测试 A：确定性数据上的过拟合 —— 验证整条链路（编码器→denoiser→采样→反归一化）接好了。
测试 B：双模态数据上 BC vs FM —— 这是 FM 值得用的根本原因。

测试 B 的构造：同一个观测下，专家有两种同样合法的动作（模态 +1 和 -1，各占一半）。
  BC 用 MSE 回归 → 收敛到条件均值 0 → 输出一个两种模态都不是的非法动作
  FM 是生成式的  → 应当能采出两个模态
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.denoiser import FMDenoiser  # noqa: E402
from src.models.policy import BCPolicy, FMPolicy  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)


# ---------------------------------------------------------------- 测试 A
def test_overfit():
    print("=== 测试 A：端到端过拟合（确定性数据）===")
    N, H, A = 32, 16, 7
    instructions = ["Pick up the red cube", "Stack cube A on cube B", "Insert the peg into the hole"]
    batch = {
        "rgb": torch.randn(N, 2, 3, 126, 126, device=DEV),
        "proprio": torch.randn(N, 2, 9, device=DEV),
        "action": torch.randn(N, H, A, device=DEV) * 0.5,
        "instruction": [instructions[i % 3] for i in range(N)],
    }

    policy = FMPolicy(cfg_dropout=0.0).to(DEV)
    policy.normalizer.fit(batch["action"])
    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=3e-4)

    policy.train()
    for step in range(601):
        loss = policy.compute_loss(batch)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"    step {step:4d}  loss {loss.item():.4f}")

    policy.eval()
    pred = policy.predict_action(batch["rgb"], batch["proprio"], batch["instruction"], n_steps=20)
    err = (pred - batch["action"]).abs().mean().item()
    scale = batch["action"].abs().mean().item()
    print(f"    采样重建误差 {err:.4f} (动作尺度 {scale:.4f}, 相对 {100*err/scale:.1f}%)")
    print("    " + ("OK — 链路打通，能拟合确定性映射" if err < 0.3 * scale else "!! 误差偏大，需排查"))
    return err < 0.3 * scale


# ---------------------------------------------------------------- 测试 B
def make_bimodal(n, H=16, A=7):
    """同一观测下的双模态动作：一半 +1，一半 -1。"""
    sign = torch.where(torch.rand(n, device=DEV) < 0.5, 1.0, -1.0)
    return sign[:, None, None] * torch.ones(n, H, A, device=DEV) + 0.05 * torch.randn(n, H, A, device=DEV)


def test_multimodal():
    print("\n=== 测试 B：双模态数据上 BC vs FM ===")
    H, A, d = 16, 7, 256
    # 观测固定 → context 固定，直接用一个常量 context，把问题隔离到建模能力本身
    context = torch.randn(1, 83, d, device=DEV) * 0.5

    # ---- FM ----
    fm = FMDenoiser(d_model=d, d_context=d, act_dim=A, act_horizon=H).to(DEV)
    opt = torch.optim.AdamW(fm.parameters(), lr=3e-4)
    for step in range(3000):
        x1 = make_bimodal(256)
        x0 = torch.randn_like(x1)
        t = torch.rand(256, device=DEV)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        loss = F.mse_loss(fm(xt, t, context.expand(256, -1, -1)), x1 - x0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"    FM 训练完毕，最终 loss {loss.item():.4f}")

    fm.eval()
    with torch.no_grad():
        x = torch.randn(2000, H, A, device=DEV)
        for i in range(20):
            t = torch.full((2000,), i / 20, device=DEV)
            x = x + fm(x, t, context.expand(2000, -1, -1)) / 20
    fm_samples = x.mean(dim=(1, 2))          # 每个样本压成一个标量便于看分布

    # ---- BC（同样的数据，MSE 回归）----
    bc = torch.nn.Sequential(
        torch.nn.Linear(d, 512), torch.nn.SiLU(),
        torch.nn.Linear(512, 512), torch.nn.SiLU(),
        torch.nn.Linear(512, H * A),
    ).to(DEV)
    opt = torch.optim.AdamW(bc.parameters(), lr=3e-4)
    ctx_pooled = context.mean(dim=1)
    for step in range(3000):
        x1 = make_bimodal(256)
        pred = bc(ctx_pooled.expand(256, -1)).view(256, H, A)
        loss = F.mse_loss(pred, x1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"    BC 训练完毕，最终 loss {loss.item():.4f}")
    with torch.no_grad():
        bc_samples = bc(ctx_pooled.expand(2000, -1)).view(2000, H, A).mean(dim=(1, 2))

    # ---- 判据：样本落在哪个模态附近 ----
    def mode_stats(s, name):
        near_pos = (s - 1.0).abs() < 0.3
        near_neg = (s + 1.0).abs() < 0.3
        covered = (near_pos | near_neg).float().mean().item()
        print(f"    {name}: 落在模态附近的比例 {100*covered:5.1f}%  "
              f"(+1模态 {100*near_pos.float().mean():4.1f}% / "
              f"-1模态 {100*near_neg.float().mean():4.1f}%)  "
              f"均值 {s.mean():+.3f}  标准差 {s.std():.3f}")
        return covered

    print()
    fm_cov = mode_stats(fm_samples, "FM")
    bc_cov = mode_stats(bc_samples, "BC")

    # ---- 作图 ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, s, name, color in ((a1, fm_samples, "Flow Matching", "#378ADD"),
                               (a2, bc_samples, "BC (MSE regression)", "#C4682B")):
        ax.hist(s.cpu().numpy(), bins=80, range=(-2, 2), color=color, alpha=0.85)
        for m in (-1, 1):
            ax.axvline(m, ls="--", c="#2A9D5C", lw=1.5)
        ax.axvline(0, ls=":", c="#999", lw=1.5)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("action value")
    a1.set_ylabel("count")
    a1.text(0.02, 0.95, "green dashed = the two valid modes\ngrey dotted = their mean (invalid)",
            transform=a1.transAxes, fontsize=8, va="top")
    plt.tight_layout()
    out = str(Path(__file__).parent / "w1_multimodal.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n    图已保存: {out}")
    return fm_cov, bc_cov


if __name__ == "__main__":
    ok = test_overfit()
    fm_cov, bc_cov = test_multimodal()
    print("\n" + "=" * 60)
    print(f"结论：FM 覆盖模态 {100*fm_cov:.1f}%，BC 覆盖 {100*bc_cov:.1f}%")
    print("BC 被 MSE 拉向条件均值，输出的是两个模态之间的非法动作。")
    print("=" * 60)
