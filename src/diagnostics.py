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
    if getattr(dataset, "feature_cache", None) is not None:
        # 走缓存时 "rgb" 是 (T, n_patches, 384) 的冻结特征，不是像素，
        # 拿 ImageNet 的均值方差去判它没有意义。改为检查缓存本身是不是活的：
        # 全零或含 NaN 说明缓存建坏了，而那种错误在 loss 上要好几千步才看得出来。
        assert rgb.ndim == 3, f"缓存特征应为 (T,N,D)，得到 {tuple(rgb.shape)}"
        assert torch.isfinite(rgb).all(), "缓存特征里有 NaN/Inf"
        assert rgb.abs().max() > 0, "缓存特征全零，缓存八成建坏了"
        if verbose:
            print(f"  冻结特征缓存: {tuple(rgb.shape)}  "
                  f"mean {rgb.mean():+.3f} std {rgb.std():.3f}", flush=True)
        return snr
    assert rgb.ndim == 4, f"rgb 应为 (T,3,H,W)，得到 {tuple(rgb.shape)}"
    if verbose:
        print(f"  图像归一化后: mean {rgb.mean():+.3f} std {rgb.std():.3f} "
              f"(ImageNet 归一化后应在 mean≈0±0.5, std≈1±0.4 范围)", flush=True)
    return snr


def _ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    """岭回归。D > N 时走对偶形式，解 N×N 而不是 D×D 的方程。

    patch token 展平后有 81×384 ≈ 31k 维，远多于样本数，primal 形式在这个规模上
    既慢又病态。两条路数学上等价。
    """
    y_mu = Ytr.mean(0)
    Yc = Ytr - y_mu
    if Xtr.shape[1] > len(Xtr):
        K = Xtr @ Xtr.T
        A = np.linalg.solve(K + alpha * np.eye(len(K)), Yc)
        return (Xte @ Xtr.T) @ A + y_mu
    D = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(D), Xtr.T @ Yc)
    return Xte @ W + y_mu


def linear_probe(features: np.ndarray, targets: np.ndarray, train_frac: float = 0.7,
                 alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6), seed: int = 0) -> dict:
    """岭回归探针：冻结特征里到底还有没有这个量。

    三个设计点，缺一个结论就不成立：

    1. **在留出集上报 R²，不是训练集。** patch token 有 31k 维，训练集上拟合
       任意 300 个样本都能得到 R²≈1，看不出任何东西。
    2. **基准用训练集均值**，所以 R² 可以为负。用留出集自己的均值做基准是作弊的，
       那样 R² 永远非负，也就永远看不出"特征里根本没有这个量"。
    3. **正则强度在内层验证集上选**，不在报告用的留出集上选。这是给"信息存在"
       这一侧最好的机会 —— 否则一个欠正则的探针会把任何东西都测成不可观测，
       负结论就没有意义了。

    features (N, D)   targets (N, k)
    返回 {"r2": 逐轴 R², "rmse": 逐轴 RMSE, "target_std": 目标自身标准差, "alpha": 选中的正则}
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    n_tr = int(len(features) * train_frac)
    tr, te = perm[:n_tr], perm[n_tr:]

    X, Y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd

    # 内层再切一刀选 alpha，报告用的留出集全程不参与
    n_fit = int(len(tr) * 0.75)
    fit, val = slice(0, n_fit), slice(n_fit, len(tr))
    best, best_alpha = -np.inf, alphas[0]
    for a in alphas:
        pred = _ridge_fit_predict(Xtr[fit], Y[tr][fit], Xtr[val], a)
        base = Y[tr][val] - Y[tr][fit].mean(0)
        r2 = 1.0 - ((pred - Y[tr][val]) ** 2).sum(0) / np.maximum((base ** 2).sum(0), 1e-12)
        if r2.mean() > best:
            best, best_alpha = r2.mean(), a

    pred = _ridge_fit_predict(Xtr, Y[tr], Xte, best_alpha)
    err = pred - Y[te]
    base = Y[te] - Y[tr].mean(0)
    return {
        "r2": 1.0 - (err ** 2).sum(0) / np.maximum((base ** 2).sum(0), 1e-12),
        "rmse": np.sqrt((err ** 2).mean(0)),
        "target_std": Y.std(0),
        "alpha": best_alpha, "n_train": len(tr), "n_test": len(te),
    }
