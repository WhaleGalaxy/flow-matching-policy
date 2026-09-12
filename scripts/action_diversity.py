"""这个任务的动作分布到底是不是多模态的？

用生成模型做策略的理由是：同一个观测对应多种合法动作，MSE 回归收敛到条件均值，
而均值可能是非法动作。`notebooks/w1_overfit_and_multimodal.py` 在一个人造的双峰
目标上验证过这一点。但**那个论证是否适用于本项目的真实数据，是另一个问题** ——
如果 PickCube 的演示来自运动规划器、并且目标已经接进了观测，那么给定观测的动作
分布可能近乎单峰，FM 相对 BC 的这条优势就不成立。

所以这里直接量：对同一个观测采 K 个动作块，看它们散得有多开。

    python scripts/action_diversity.py --fm outputs/fmgoal_PickCube_s42/ckpt_20000.pt \
        --bc outputs/bcgoal_PickCube_s42/ckpt_20000.pt

三个尺度放在一起才有意义（全部换算到动作归一化单位）：

  * FM 的**条件**散度 —— 同一观测下 K 个样本的展开程度
  * 数据的**边缘**散度 —— 整个数据集动作的标准差，归一化后按定义是 1
  * BC 与 FM 样本均值的距离 —— 若 FM 确实在建模一个多模态分布而 BC 塌到均值，
    这个距离应当远小于 FM 自己的条件散度（BC ≈ 均值，就落在样本云的中心）

判读：条件散度 ≪ 边缘散度 → 给定观测后动作几乎确定，多模态论证在这份数据上
不成立，FM 与 BC 的成功率差距（如果有）来自别的原因，不能归给"BC 塌到均值"。

**一个必须先排除的解释**：采样散度也可能只是 Euler 积分误差，与模型学到的分布
无关。判据是看散度随采样步数怎么变 —— 积分误差会随步数增加而收缩，而模型真正
表示的条件不确定性不会。所以默认在几个步数下各测一遍。

还要注意这量的是**模型的**采样散度，不是数据真实的条件分布。两者一致需要模型
本身校准得好，本项目没有独立验证这一点。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.data.dataset import ManiskillDataset, collate, default_h5_path  # noqa: E402
from run_eval import load_policy  # noqa: E402


@torch.no_grad()
def sample_spread(policy, batch, k: int, n_steps: int, device: str):
    """同一批观测重复采样 K 次，返回逐样本的条件散度（归一化动作单位）。

    散度定义为动作块上逐元素标准差再对 (时间, 维度) 取均方根 —— 就是"K 个样本
    在动作空间里的典型半径"。归一化统计量取自 policy 自己的 normalizer，
    所以和数据的边缘标准差（按定义为 1）直接可比。
    """
    rgb, prop = batch["rgb"].to(device), batch["proprio"].to(device)
    inst = batch["instruction"]
    B = rgb.shape[0]

    samples = []
    for _ in range(k):
        a = policy.predict_action(rgb, prop, inst, n_steps=n_steps, guidance=1.0)
        samples.append(policy.normalizer.normalize(a))
    s = torch.stack(samples)                          # (K, B, T, A)
    per_elem_std = s.std(dim=0)                       # (B, T, A)
    spread = per_elem_std.pow(2).mean(dim=(1, 2)).sqrt()   # (B,)
    return spread.float().cpu().numpy(), s.mean(0)    # 散度, 样本均值(归一化)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", type=Path, required=True)
    ap.add_argument("--bc", type=Path, default=None)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n-obs", type=int, default=256, help="抽多少个观测")
    ap.add_argument("--k", type=int, default=32, help="每个观测采多少个动作块")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-steps", type=int, nargs="+", default=[2, 10, 50],
                    help="在这几个采样步数下各测一遍，用来区分"
                         "\"模型表示的不确定性\"和\"积分误差\"")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fig", type=Path, default=Path("docs/figures/action_diversity.png"))
    args = ap.parse_args()

    fm, cfg, name = load_policy(args.fm, use_ema=False, device=args.device)
    assert name == "fm", f"--fm 指向的是 {name} 的 checkpoint"
    bc = None
    if args.bc:
        bc, bcfg, bname = load_policy(args.bc, use_ema=False, device=args.device)
        assert bname.startswith("bc"), f"--bc 指向的是 {bname} 的 checkpoint"
        # 观测配置必须逐项一致。只比 use_goal 是不够的：现在有三种目标接法
        # （use_goal / goal_slot / task_goal），任意一项不同，两个策略看到的
        # proprio 就不是同一个东西，"BC 的预测离 FM 的样本云多远"这个量也就没有意义。
        obs_keys = ("use_goal", "goal_slot", "task_goal", "use_proprio")
        assert all(bcfg.get(k) == cfg.get(k) for k in obs_keys), (
            f"两个 checkpoint 的观测配置不同，比较没有意义：\n"
            f"  fm: {[(k, cfg.get(k)) for k in obs_keys]}\n"
            f"  bc: {[(k, bcfg.get(k)) for k in obs_keys]}")

    ds = ManiskillDataset(default_h5_path(args.task), task=args.task,
                          obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
                          img_size=cfg["img_size"], use_goal=cfg.get("use_goal", False),
                          goal_slot=cfg.get("goal_slot", False),
                          task_goal=cfg.get("task_goal", False))
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(args.n_obs, len(ds)), replace=False)

    by_steps, bc_gaps = {}, []
    for n_steps in args.n_steps:
        chunks = []
        for i in range(0, len(picks), args.batch_size):
            batch = collate([ds[int(j)] for j in picks[i:i + args.batch_size]])
            sp, mean_norm = sample_spread(fm, batch, args.k, n_steps, args.device)
            chunks.append(sp)
            # BC 的对照只在主步数下算一次，它本来就与采样步数无关
            if bc is not None and n_steps == args.n_steps[-1]:
                pred = bc.predict_action(batch["rgb"].to(args.device),
                                         batch["proprio"].to(args.device),
                                         batch["instruction"])
                gap = (bc.normalizer.normalize(pred) - mean_norm)
                bc_gaps.append(gap.pow(2).mean(dim=(1, 2)).sqrt().float().cpu().numpy())
        by_steps[n_steps] = np.concatenate(chunks)

    main_steps = args.n_steps[-1]
    spreads = by_steps[main_steps]
    bc_gaps = np.concatenate(bc_gaps) if bc_gaps else None

    print(f"\n===== 动作多样性：{args.task}"
          f"（{len(spreads)} 个观测 × {args.k} 次采样）=====")
    print(f"  数据的边缘散度（动作归一化后按定义）  1.000\n")
    print(f"  {'采样步数':<10}{'条件散度中位数':>16}{'占边缘的比例':>14}")
    for n_steps, sp in by_steps.items():
        print(f"  {n_steps:<10}{np.median(sp):>16.3f}{100*np.median(sp):>13.0f}%")

    # 收敛性看最后两个步数：还在明显变化说明步数不够，读到的不是模型的分布
    if len(args.n_steps) >= 2:
        prev, last = (np.median(by_steps[args.n_steps[-2]]), np.median(spreads))
        tail = abs(last - prev) / max(prev, 1e-9)
        lo = np.median(by_steps[args.n_steps[0]])
        print(f"\n  {args.n_steps[-2]} → {args.n_steps[-1]} 步之间还变化 "
              f"{100*tail:.0f}%", end="")
        if tail < 0.15:
            print(" —— 已收敛，这个散度是模型学到的条件分布，不是积分误差。")
        else:
            print(" —— 尚未收敛，再加步数还会变，别把它当成模型的分布。")
        if last > lo * 1.2:
            print(f"  少步采样**低估**散度：{args.n_steps[0]} 步只有 {lo:.3f}，"
                  f"{args.n_steps[-1]} 步是 {last:.3f}。")
            print("  一两步 Euler 约等于沿平均速度走一大步，会把样本云压向条件均值 ——")
            print("  也就是说，**步数少的 Flow Matching 行为上更接近 BC**。")
            print("  推论：若这个任务的条件分布本来就窄，少步就不该有损失，")
            print("  成功率-步数曲线会是平的。这一点由主图独立检验。")

    if bc_gaps is not None:
        print(f"\n  BC 预测与 FM 样本均值的距离  中位数 {np.median(bc_gaps):.3f}")
        print(f"    （若 BC 塌到条件均值而 FM 建模了多模态，这个距离应当"
              f"明显小于 FM 的条件散度 {hi:.3f}）")

    print(f"\n  注意：这量的是**模型的**采样散度，不是数据真实的条件分布 —— "
          "\n        两者一致需要模型校准得好，本项目没有独立验证这一点。")

    plot(spreads, bc_gaps, args.fig, args.task, args.k, main_steps)
    print(f"\n已写出 {args.fig}")


def plot(spreads, bc_gaps, path: Path, task: str, k: int, n_steps: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(spreads, bins=40, color="#2f6f4e", alpha=0.85,
            label=f"FM conditional spread ({k} samples/obs, {n_steps} sampling steps)")
    if bc_gaps is not None:
        ax.hist(bc_gaps, bins=40, color="#6b6b6b", alpha=0.6,
                label="distance from BC prediction to FM sample mean")
    ax.axvline(1.0, color="#8c2d1e", lw=2, ls="--",
               label="marginal spread of the demonstrations (= 1 by normalisation)")
    ax.set_xlabel("spread in normalised action units")
    ax.set_ylabel("number of observations")
    ax.set_title(f"Is the action distribution actually multimodal? ({task})")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
