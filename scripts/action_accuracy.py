"""各方法对演示动作的复现误差 —— 把"学到的函数准不准"从 rollout 里剥出来。

成功率是闭环指标，把建模误差、积分误差、执行噪声和误差累积全混在一起。这里只问
一个开环问题：**给定观测，各方法预测的动作块离演示的真实动作块有多远。**

三条对照，全部换算到动作归一化单位（数据的边缘标准差按定义为 1）：

    BC            一次前向，输出的就是条件均值的估计
    FM 单样本     推理时实际执行的东西
    FM K 样本均值 蒙特卡洛估计的条件均值，与 BC 同性质

如果 `FM K 样本均值` 的误差明显大于 `BC`，说明差距不在"采样注入噪声"，
而在**学到的条件均值本身更差** —— 也就是同等容量和预算下，去建模整个条件分布
是有代价的，而这份数据的条件分布是单峰的（见 scripts/conditional_spread.py），
这个代价换不回任何东西。

误差在**留出集**上算：这些 episode 没有参与训练，所以报的是泛化误差。
留出的划分从 checkpoint 的 cfg 里读（val_frac / split_seed），三个方法共用同一个
split_seed，因此拿到的是逐条相同的留出集。

val_frac=0 训练出来的 checkpoint（2026-09-12 之前的全部结果）没有留出集可用，
这里会直接报错而不是安静地退回训练集 —— "泛化误差"算在训练过的数据上不会有任何
异常表现，只会给出一个偏小的数字。要复现旧结果就显式加 --split train。

    python scripts/action_accuracy.py --ckpt outputs/fmnocfg_mt_tg60_s42/ckpt_60000.pt \
                                      outputs/bcx_mt_tg60_s42/ckpt_60000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import load_policy  # noqa: E402
from src.data.dataset import ManiskillDataset, collate, default_h5_path  # noqa: E402


@torch.no_grad()
def errors(policy, batch, device, n_steps, n_average=1):
    """归一化动作空间里的逐样本 RMSE，外加夹爪落在中间带的比例。

    夹爪（最后一维）在演示里几乎总在 ±1 附近 —— 张开或闭合，中间值不是一个
    物理上有意义的指令。实测真实演示里 |g|<0.5 的比例是 **0.0%**，而把两个合法
    动作块平均一下就能造出 49.5%。所以这个比例是"预测的动作块像不像真的"的
    一个直接判据，且与 RMSE 互补：RMSE 会被幅值误差主导，看不出这种结构性错误。
    """
    pred = policy.predict_action(batch["rgb"].to(device), batch["proprio"].to(device),
                                 batch["instruction"], n_steps=n_steps,
                                 guidance=1.0, n_average=n_average)
    p = policy.normalizer.normalize(pred)
    t = policy.normalizer.normalize(batch["action"].to(device))
    rmse = (p - t).pow(2).mean(dim=(1, 2)).sqrt().float().cpu().numpy()
    g = pred[..., -1]                                   # 原始单位，±1 是物理量
    mid = ((g > -0.5) & (g < 0.5)).float().mean(dim=1).cpu().numpy()
    return rmse, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n-obs", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k", type=int, default=16, help="FM 取均值时用几个样本")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="在哪个划分上算误差。默认 val（泛化误差）")
    args = ap.parse_args()

    split_label = {"val": "留出集，泛化误差", "train": "训练集，拟合误差",
                   "all": "全部数据"}[args.split]
    print(f"\n===== 动作复现误差：{args.task}（{args.n_obs} 个观测，{split_label}）=====")
    print("归一化动作单位，数据的边缘标准差按定义为 1.000\n")
    print(f"  {'方法':<34}{'RMSE 中位数':>14}{'占边缘':>10}{'夹爪中间带':>12}")

    for ck in args.ckpt:
        policy, cfg, name = load_policy(ck, use_ema=False, device=args.device)
        val_frac = float(cfg.get("val_frac", 0.0))
        assert not (args.split == "val" and val_frac <= 0), (
            f"{ck} 是 val_frac=0 训练的（全部 episode 都进了训练），没有留出集。\n"
            f"要么用留出集重训，要么显式 --split train 并把结果标成拟合误差。")
        ds = ManiskillDataset(
            default_h5_path(args.task), task=args.task,
            obs_horizon=cfg["obs_horizon"], act_horizon=cfg["act_horizon"],
            img_size=cfg["img_size"], use_goal=cfg.get("use_goal", False),
            goal_slot=cfg.get("goal_slot", False), task_goal=cfg.get("task_goal", False),
            val_frac=val_frac, split_seed=int(cfg.get("split_seed", 0)),
            split=args.split)
        print(f"  [{name}] {args.split} 划分：{len(ds.episode_lengths)} 条轨迹 / "
              f"{len(ds):,} 个样本", flush=True)
        rng = np.random.default_rng(0)
        picks = rng.choice(len(ds), size=min(args.n_obs, len(ds)), replace=False)
        steps = cfg["model"].get("n_sample_steps", 1 if name.startswith("bc") else 10)

        variants = [("", 1)] if name.startswith("bc") else [
            ("（单样本，实际推理用的）", 1), (f"（{args.k} 样本均值）", args.k)]
        for label, k in variants:
            errs, mids = [], []
            for i in range(0, len(picks), args.batch_size):
                b = collate([ds[int(j)] for j in picks[i:i + args.batch_size]])
                r, m = errors(policy, b, args.device, steps, k)
                errs.append(r); mids.append(m)
            e = np.median(np.concatenate(errs))
            mid = np.mean(np.concatenate(mids))
            tag = f"{name}{label}"
            print(f"  {tag:<34}{e:>14.3f}{100*e:>9.0f}%{100*mid:>11.1f}%")
        del policy
        torch.cuda.empty_cache()

    # 真实演示的夹爪中间带比例，作为参照
    ds0 = ManiskillDataset(default_h5_path(args.task), task=args.task,
                           val_frac=float(cfg.get("val_frac", 0.0)),
                           split_seed=int(cfg.get("split_seed", 0)), split=args.split)
    rng = np.random.default_rng(0)
    gs = np.concatenate([ds0[int(j)]["action"][:, -1].numpy()
                         for j in rng.choice(len(ds0), 2000, replace=False)])
    print(f"\n  {'真实演示（参照）':<34}{'—':>14}{'—':>10}"
          f"{100*float(((gs > -0.5) & (gs < 0.5)).mean()):>11.1f}%")
    print("\n判读：若 FM 的『K 样本均值』误差仍明显大于 BC，说明差距不在采样噪声，"
          "\n      而在学到的条件均值本身 —— 同等容量下建模整个条件分布是有代价的。"
          "\n      夹爪中间带那一列量的是另一件事：预测出来的动作块结构上像不像真的。"
          "\n      夹爪本质是离散的，用高斯流或 MSE 去建它，中间带就会被填上。")


if __name__ == "__main__":
    main()
