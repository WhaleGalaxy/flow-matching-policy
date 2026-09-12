"""策略的输出到底有没有在跟着观测变？

用来区分两件在成功率上长得一模一样、但下一步完全不同的事：

  * 策略学到了任务但做得不好  → 输出随观测变化，只是不够准
  * 策略根本没在用观测        → 输出近乎常数，成功率等于"闭着眼睛做"的水平

后者的现场指纹是**各种配置下的评测结果精确相等**：改采样步数、改执行长度，
平均步长纹丝不动。本项目撞到过两次（见 docs/debugging.md），所以固化成脚本。

    python scripts/check_policy_output.py --ckpt outputs/fmgoal_.../ckpt_20000.pt \
        outputs/ddpmgoal_.../ckpt_20000.pt outputs/bcgoal_.../ckpt_20000.pt

三个数放在一起读（全部在动作归一化单位下，所以数据本身的尺度是 1）：

  * **跨观测的标准差** —— 换一个观测，输出跟着变多少。≈0 就是没在看观测。
  * **对真值的 MSE** —— 与演示动作的距离。
  * **同一量的基准：预测数据集均值**。归一化后这个基准的 MSE 恰好是 1.0，
    所以 MSE ≈ 1.0 意味着"还不如不学"，MSE < 1.0 才说明学到了东西。
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
def probe(policy, ds, picks, batch_size, device, n_steps=None):
    preds, truths = [], []
    for i in range(0, len(picks), batch_size):
        batch = collate([ds[int(j)] for j in picks[i:i + batch_size]])
        kw = {} if n_steps is None else {"n_steps": n_steps}
        a = policy.predict_action(batch["rgb"].to(device), batch["proprio"].to(device),
                                  batch["instruction"], **kw)
        preds.append(policy.normalizer.normalize(a).float().cpu())
        truths.append(policy.normalizer.normalize(batch["action"].to(device))
                      .float().cpu())
    pred, truth = torch.cat(preds), torch.cat(truths)

    # 跨观测的变化量：先在样本维求标准差，再对 (时间, 动作维) 取均方根
    across_obs = pred.std(dim=0).pow(2).mean().sqrt().item()
    truth_across = truth.std(dim=0).pow(2).mean().sqrt().item()
    mse = (pred - truth).pow(2).mean().item()
    # "永远输出数据集均值"的基准。归一化后按定义就是 1.0，这里实测一下确认
    baseline = truth.pow(2).mean().item()
    return dict(across_obs=across_obs, truth_across=truth_across,
                mse=mse, baseline=baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n-obs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-x0", type=float, default=None,
                    help="覆盖 DDPM 的 x0 裁剪阈值。传 0 关掉裁剪，用来复现"
                         "未修复时的行为（见 DDPMPolicy.predict_action）")
    args = ap.parse_args()

    print(f"\n{'checkpoint':<34}{'跨观测 std':>12}{'对真值 MSE':>12}"
          f"{'预测均值基准':>14}  判读")
    print("-" * 92)
    for ckpt in args.ckpt:
        policy, cfg, name = load_policy(ckpt, use_ema=False, device=args.device)
        if args.clip_x0 is not None and hasattr(policy, "clip_x0"):
            policy.clip_x0 = args.clip_x0
        ds = ManiskillDataset(default_h5_path(args.task), task=args.task,
                              obs_horizon=cfg["obs_horizon"],
                              act_horizon=cfg["act_horizon"], img_size=cfg["img_size"],
                              use_goal=cfg.get("use_goal", False))
        picks = np.random.default_rng(0).choice(
            len(ds), size=min(args.n_obs, len(ds)), replace=False)
        r = probe(policy, ds, picks, args.batch_size, args.device)

        if r["across_obs"] < 0.1 * r["truth_across"]:
            verdict = "❌ 输出几乎不随观测变化 —— 没在用观测"
        elif r["mse"] >= r["baseline"]:
            verdict = "❌ 不如直接输出数据集均值"
        elif r["mse"] > 0.6 * r["baseline"]:
            verdict = "⚠️  比均值好，但差得不多"
        else:
            verdict = "✅ 输出跟着观测走，且明显优于均值"
        tag = ckpt.parent.name + "/" + name
        if args.clip_x0 is not None and hasattr(policy, "clip_x0"):
            tag += f" clip={args.clip_x0:g}"
        print(f"{tag:<34}{r['across_obs']:>12.3f}"
              f"{r['mse']:>12.3f}{r['baseline']:>14.3f}  {verdict}")
        del policy
        torch.cuda.empty_cache()

    print("-" * 92)
    print(f"参考：演示动作本身的跨观测 std = {r['truth_across']:.3f}"
          f"（归一化后应接近 1）")
    print("注：这里量的是**单步开环预测**，不是闭环成功率。开环准不代表闭环成功，")
    print("    但开环等于常数，闭环一定不可能成功 —— 这是个必要条件检查。")


if __name__ == "__main__":
    main()
