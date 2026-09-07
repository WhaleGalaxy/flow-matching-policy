"""训练前的健全性检查。

这些检查全部来自实际踩过的坑（见 docs/debugging.md）。它们很便宜（几秒），
但能在浪费 45 分钟训练之前拦住整类错误。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def action_representation_snr(dataset, n_batches: int = 8, batch_size: int = 512,
                              num_workers: int = 4) -> dict:
    """量化动作表示的信噪比。

    做法：用当前本体状态（qpos）**线性**回归动作块，看能解释掉多少动作方差。
    能被解释掉的部分只是在复述"机器人当前在哪"，不含任务信息；剩下的残差才是
    策略真正需要从视觉/语言里学的东西。

    实测：
      pd_joint_pos（绝对关节角）  → 98.4% 可解释，任务信号仅 1.6%，策略学不出来
      pd_ee_delta_pose（末端增量）→ 36.1% 可解释，任务信号 63.9%，可用

    返回 {"explained": 可解释比例, "signal": 任务信号比例}
    """
    from src.data.dataset import collate

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, collate_fn=collate)
    A, P = [], []
    for i, b in enumerate(dl):
        A.append(b["action"]); P.append(b["proprio"][:, -1, :])
        if i + 1 >= n_batches:
            break
    A = torch.cat(A).flatten(1).numpy()          # (N, H*act_dim)
    P = torch.cat(P).numpy()                      # (N, proprio_dim)

    An = (A - A.mean(0)) / (A.std(0) + 1e-8)
    Pc = np.concatenate([P - P.mean(0), np.ones((len(P), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Pc, An, rcond=None)
    residual = float(((An - Pc @ W) ** 2).mean())
    return {"explained": 1.0 - residual, "signal": residual}


def check_dataset(dataset, verbose: bool = True) -> dict:
    """跑一遍所有检查，返回结果；有问题时打印醒目警告。"""
    snr = action_representation_snr(dataset)
    if verbose:
        print(f"  动作表示信噪比: 任务信号 {100*snr['signal']:.1f}%  "
              f"(其余 {100*snr['explained']:.1f}% 仅由当前本体状态线性决定)", flush=True)
        if snr["signal"] < 0.15:
            print("  " + "!" * 60, flush=True)
            print("  !! 警告：任务信号占比过低，策略很可能学不出可用行为。", flush=True)
            print("  !! 绝对位置类的动作表示（如 pd_joint_pos）会有这个问题，", flush=True)
            print("  !! 请改用增量表示（pd_ee_delta_pose / pd_joint_delta_pos）。", flush=True)
            print("  !! 详见 docs/debugging.md", flush=True)
            print("  " + "!" * 60, flush=True)

    sample = dataset[0]
    rgb = sample["rgb"]
    assert rgb.ndim == 4, f"rgb 应为 (T,3,H,W)，得到 {tuple(rgb.shape)}"
    if verbose:
        print(f"  图像归一化后: mean {rgb.mean():+.3f} std {rgb.std():.3f} "
              f"(ImageNet 归一化后应在 mean≈0±0.5, std≈1±0.4 范围)", flush=True)
    return snr
