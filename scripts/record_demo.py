"""录一段策略 rollout，拼成 README 顶部那张动图。

面试官在仓库上平均花不到一分钟，一张动图买的是注意力，所以这件事值得做对：

- **只收成功的那一局。** `rollout(record_frames=True)` 录的是第 0 局，
  而第 0 局可能是失败的。这里逐个 seed 试，拿到第一局成功的就停。
  展示失败的 rollout 不是诚实，是没讲清楚 —— 成功率写在旁边的表里，
  动图要回答的是"它到底在做什么"。
- **三个任务并排。** 单看一个任务看不出这是多任务策略，而多任务正是主张的一部分。
- **控制体积。** GitHub 上超过几 MB 的 GIF 会加载很慢，很多人根本等不到它动起来。
  所以默认抽帧 + 缩放，成品目标 2MB 以内。

    python scripts/record_demo.py --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt \
        --out docs/figures/demo.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def compose(panels: list[np.ndarray], labels: list[str], scale: int = 192,
            stride: int = 3) -> np.ndarray:
    """把每个任务的一段帧序列横向拼成一段动画。

    panels[i] 形状 (T_i, H, W, 3) uint8，长度可以不同：短的用最后一帧补齐，
    于是先完成的任务停在成功姿态上，而不是凭空消失或循环重放。
    """
    assert panels and len(panels) == len(labels)
    from PIL import Image, ImageDraw

    T = max(len(p) for p in panels)
    idx = list(range(0, T, stride))
    if idx[-1] != T - 1:
        idx.append(T - 1)

    out = []
    for t in idx:
        row = []
        for p, lab in zip(panels, labels):
            f = p[min(t, len(p) - 1)]
            im = Image.fromarray(np.asarray(f, dtype=np.uint8)).resize(
                (scale, scale), Image.BILINEAR)
            d = ImageDraw.Draw(im)
            # 先描一圈深色再写浅色，避免标签落在浅色桌面上看不见
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                d.text((6 + dx, 6 + dy), lab, fill=(0, 0, 0))
            d.text((6, 6), lab, fill=(255, 255, 255))
            row.append(np.asarray(im))
        out.append(np.concatenate(row, axis=1))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+",
                    default=["PickCube-v1", "PushCube-v1", "StackCube-v1"])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/demo.gif")
    ap.add_argument("--max-tries", type=int, default=12,
                    help="每个任务最多试几个 seed 去找一局成功的")
    ap.add_argument("--scale", type=int, default=192)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import torch
    from run_eval import load_policy
    from src.evaluate import make_env, obs_kwargs, rollout

    policy, cfg, name = load_policy(args.ckpt, use_ema=False, device=args.device)
    n_steps = cfg["model"].get("n_sample_steps", 1 if name.startswith("bc") else 10)

    panels, labels = [], []
    for task in args.tasks:
        env = make_env(task)
        best, best_frames = None, None
        try:
            for seed in range(args.max_tries):
                r = rollout(policy, env, task, n_episodes=1,
                            obs_horizon=cfg["obs_horizon"],
                            execute_horizon=cfg["eval"]["execute_horizon"],
                            img_size=cfg["img_size"], n_steps=n_steps, guidance=1.0,
                            max_steps=args.max_steps, seed=seed, record_frames=True,
                            **obs_kwargs(cfg))
                ok = r["success_rate"] > 0.5
                print(f"  {task:16s} seed={seed:<3} {'成功' if ok else '失败'} "
                      f"({int(r['mean_length'])} 步)", flush=True)
                if ok:
                    best, best_frames = seed, r["frames"]
                    break
                if best_frames is None:          # 一局都没成时留最后一次当兜底
                    best_frames = r["frames"]
        finally:
            env.close()
        if best is None:
            print(f"  ! {task} 在 {args.max_tries} 个 seed 里没有成功的，"
                  f"动图里这一格是失败的一局", flush=True)
        panels.append(np.stack(best_frames))
        labels.append(task.replace("-v1", ""))
        del env
        torch.cuda.empty_cache()

    anim = compose(panels, labels, scale=args.scale, stride=args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, list(anim), fps=args.fps, loop=0)
    mb = args.out.stat().st_size / 1e6
    print(f"\n写出 {args.out}  {anim.shape[0]} 帧  {mb:.1f} MB")
    if mb > 4:
        print("  体积偏大，调大 --stride 或调小 --scale 再录一次")


if __name__ == "__main__":
    main()
