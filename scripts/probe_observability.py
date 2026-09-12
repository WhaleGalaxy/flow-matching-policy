"""观测充分性检查：任务里的每个关键物理量，能不能从策略实际看到的观测里恢复出来。

这是本项目代价最高的一课变成的工具。PickCube 上策略的抓取率 95%、成功率却只有
6%，排查了很久才发现目标位置在 `base_camera` 的图像里根本不存在 —— 每局随机的
目标是一个半透明标记，模型被要求预测一个依赖于它无法观测的量的动作。
接进 proprio 之后成功率 6% → 75%。

同样的问题，用这个脚本几分钟就能在训练之前答完：

    python scripts/probe_observability.py --task PickCube-v1
    python scripts/probe_observability.py --task StackCube-v1 --camera hand_camera

判读：R² 接近 1 → 信息在观测里，模型学不到只能怪模型；R² 接近 0 或为负 →
信息不在观测里，换模型、加训练步数、调超参全都没用，只能改观测。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import default_h5_path, preprocess_obs_rgb  # noqa: E402
from src.diagnostics import linear_probe  # noqa: E402
from src.models.encoders import VisualEncoder  # noqa: E402

# 桌面是静态的，探它没有意义
SKIP_ACTORS = {"table-workspace"}


def discover_targets(ep: h5py.Group) -> dict[str, tuple[str, ...]]:
    """列出这个任务里值得探的物理量：场景中每个可动物体的位置，以及显式目标。

    env_states/actors/<name> 的前 3 维是世界坐标位置（后面是四元数和速度）。
    """
    targets: dict[str, tuple[str, ...]] = {}
    for name in ep.get("env_states/actors", {}):
        if name not in SKIP_ACTORS:
            targets[f"{name} 位置"] = ("env_states/actors/" + name, "pos")
    if "goal_pos" in ep.get("obs/extra", {}):
        targets["goal_pos (显式目标)"] = ("obs/extra/goal_pos", "raw")
    return targets


@torch.no_grad()
def collect(h5_path: Path, n_samples: int, camera: str, img_size: int,
            obs_horizon: int, device: str, seed: int = 0):
    """随机抽帧，算冻结视觉特征，同时取出各个探测目标的真值。"""
    enc = VisualEncoder(d_model=64, img_size=img_size).to(device).eval()
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
        spec = discover_targets(f[names[0]])
        assert camera in f[names[0]]["obs/sensor_data"], (
            f"演示数据里没有相机 {camera}，可选：{list(f[names[0]]['obs/sensor_data'])}")

        # 每条轨迹只抽一帧，避免同一局内相邻帧几乎重复导致留出集其实是训练集
        picks = [(nm, int(rng.integers(0, f[nm]["obs/agent/qpos"].shape[0])))
                 for nm in rng.choice(names, size=min(n_samples, len(names)), replace=False)]

        feats, ys = [], {k: [] for k in spec}
        for nm, t in picks:
            ep = f[nm]
            ts = [max(0, t - i) for i in reversed(range(obs_horizon))]
            frames = np.stack([ep["obs/sensor_data"][camera]["rgb"][i] for i in ts])
            x = preprocess_obs_rgb(frames, img_size).unsqueeze(0).to(device)
            feats.append(enc.dino.forward_features(x[0])[:, enc.dino.num_prefix_tokens:]
                         .mean(0).flatten().float().cpu().numpy())
            for key, (path, kind) in spec.items():
                v = np.asarray(ep[path][t])
                ys[key].append(v[:3] if kind == "pos" else v)

    return np.stack(feats), {k: np.stack(v) for k, v in ys.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--img-size", type=int, default=126)
    ap.add_argument("--obs-horizon", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    feats, ys = collect(default_h5_path(args.task), args.n_samples, args.camera,
                        args.img_size, args.obs_horizon, args.device)

    print(f"\n===== 观测充分性：{args.task} / {args.camera} =====")
    print(f"冻结 DINOv2 patch token {feats.shape[1]} 维，{feats.shape[0]} 个样本"
          f"（每条轨迹抽一帧）\n")
    print(f"{'物理量':<24}{'R² (x/y/z)':<30}{'RMSE mm (x/y/z)':<26}{'该量自身 std mm'}")
    print("-" * 100)
    for key, y in ys.items():
        r = linear_probe(feats, y)
        r2 = "/".join(f"{v:+.2f}" for v in r["r2"])
        rmse = "/".join(f"{1000*v:.0f}" for v in r["rmse"])
        std = "/".join(f"{1000*v:.0f}" for v in r["target_std"])
        print(f"{key:<24}{r2:<30}{rmse:<26}{std}")
    print("-" * 100)
    print("判读：R²→1 信息在观测里，学不到是模型的问题；R²≤0 信息不在观测里，")
    print("      换模型/加步数/调超参都无效，只能改观测（接入该量，或换能看见它的相机）。")


if __name__ == "__main__":
    main()
