"""Week0 Day3 补充：FM 的 loss 有不可约下界，绝对值不能用来判断训练好坏。

原因：给定 (x_t, t)，回归目标 x1-x0 仍然是**随机的**。
  t=0 时 x_t=x0，此时 x1 完全未知 → 目标方差 = 数据方差
  t=1 时 x_t=x1，此时 x0 完全未知 → 目标方差 = 噪声方差 = 1
最优模型输出的是条件均值 E[x1-x0 | x_t, t]，残差方差扔不掉。
所以 loss 收敛到 E[Var(x1-x0 | x_t,t)]，是个 O(1) 的正数，**不会趋近 0**。

本脚本同时记录 loss 和真实生成质量（切片 W2），看它们的走势。
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from w0d3_swiss_roll_fm import DEV, VelocityField, sample, sliced_w2, swiss_roll

torch.manual_seed(0)


def main():
    data = swiss_roll(20000).to(DEV)
    holdout = swiss_roll(4000).to(DEV)
    model = VelocityField().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    hist = {"step": [], "loss": [], "w2": []}
    bs, total, every = 512, 12000, 500
    running = []

    for step in range(total + 1):
        x1 = data[torch.randint(len(data), (bs,), device=DEV)]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, 1, device=DEV)
        loss = (model((1 - t) * x0 + t * x1, t) - (x1 - x0)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())

        if step % every == 0:
            w2 = sliced_w2(sample(model, 4000, 20), holdout)
            hist["step"].append(step)
            hist["loss"].append(sum(running) / len(running))
            hist["w2"].append(w2)
            running = []
            print(f"  step {step:6d}   loss {hist['loss'][-1]:.4f}   sliced-W2 {w2:.4f}")

    l0, l1 = hist["loss"][1], hist["loss"][-1]
    w0_, w1_ = hist["w2"][1], hist["w2"][-1]
    print(f"\nloss:  {l0:.4f} -> {l1:.4f}   （只降了 {100*(l0-l1)/l0:.1f}%）")
    print(f"W2:    {w0_:.4f} -> {w1_:.4f}   （降了 {100*(w0_-w1_)/w0_:.1f}%）")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["step"], hist["loss"], c="#C4682B", lw=2)
    a1.set_title("FM training loss — plateaus well above zero", fontsize=11)
    a1.set_xlabel("step"); a1.set_ylabel("loss"); a1.set_ylim(0, max(hist["loss"]) * 1.15)
    a1.axhline(0, ls="--", lw=1, c="#999")
    a1.grid(alpha=0.3)

    a2.plot(hist["step"], hist["w2"], c="#378ADD", lw=2)
    a2.set_title("Actual sample quality — keeps improving", fontsize=11)
    a2.set_xlabel("step"); a2.set_ylabel("sliced W2 (lower is better)")
    a2.grid(alpha=0.3)

    plt.tight_layout()
    out = "/home/xjy/flow_matching/fm_policy/notebooks/w0d3b_loss_floor.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
