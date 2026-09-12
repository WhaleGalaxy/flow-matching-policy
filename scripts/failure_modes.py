"""失败归因：策略卡在动作链条的哪一段。

一个成功率数字说不出"没学会看"和"学会了但抖"的区别，而这两者的下一步完全不同。
PickCube 的链条是 接近 → 抓取 → 送达 → 静止，环境自己就把后三段暴露成了
`is_grasped` / `is_obj_placed` / `is_robot_static`，所以分段判定不需要我自己
定阈值 —— 用的就是任务的成功判据本身拆开。

    python scripts/failure_modes.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_20000.pt
    python scripts/failure_modes.py --ckpt A.pt B.pt C.pt --labels FM DDPM BC --n-episodes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.data.dataset import (TASK_INSTRUCTIONS, build_proprio,  # noqa: E402
                              preprocess_obs_rgb)
from src.evaluate import make_env  # noqa: E402
from run_eval import load_policy  # noqa: E402

# 接近的判据用夹爪能张开的尺度，不是任务参数：方块边长 20mm
NEAR_THRESH = 0.03

# 中文用于控制台，英文用于图 —— 图要在没装中文字体的机器上也能正确渲染
STAGES = ["未接近", "接近未抓取", "抓取未送达", "送达未静止", "成功"]
STAGES_EN = ["never approached", "approached, no grasp", "grasped, not delivered",
             "delivered, not settled", "success"]
# 与 STAGES 同序：从"完全没学到"到"完全学会"，颜色也按这个梯度
STAGE_COLORS = ["#8c2d1e", "#c4622d", "#d9a441", "#6f9bd1", "#2f6f4e"]


def _b(x) -> bool:
    """ManiSkill 的标志位可能是 CUDA tensor、numpy 数组或标量。"""
    if torch.is_tensor(x):
        x = x.cpu()
    return bool(np.asarray(x).reshape(-1)[0])


def _arr(x) -> np.ndarray:
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _p3(x) -> np.ndarray:
    return _arr(x).reshape(-1)[:3]


def classify(ep: dict) -> str:
    """把一局归到链条上第一个没跨过去的环节。"""
    if ep["success"]:
        return "成功"
    if ep["placed"]:
        return "送达未静止"      # 方块到位了但机器人没停稳，success 判据不认
    if ep["grasped"]:
        return "抓取未送达"
    if ep["dmin"] <= NEAR_THRESH:
        return "接近未抓取"
    return "未接近"


@torch.no_grad()
def run(policy, env, task, cfg, n_episodes=100, n_steps=10, guidance=1.0,
        execute_horizon=None, max_steps=300, seed=0):
    device = next(policy.parameters()).device
    instruction = TASK_INSTRUCTIONS[task]
    obs_horizon = cfg["obs_horizon"]
    img_size = cfg["img_size"]
    use_goal = cfg.get("use_goal", False)
    H = execute_horizon or cfg["eval"]["execute_horizon"]
    u = env.unwrapped
    policy.eval()

    rows = []
    for e in range(n_episodes):
        # 与 src.evaluate.rollout 用同一套 seed 约定，两边的数字才能并排看
        torch.manual_seed(seed + e)
        obs, _ = env.reset(seed=seed + e)
        rgb_hist, prop_hist = deque(maxlen=obs_horizon), deque(maxlen=obs_horizon)

        def push(o):
            rgb = o["sensor_data"]["base_camera"]["rgb"][0]
            # proprio 必须与训练走同一个 build_proprio，见 src/data/dataset.py
            prop = build_proprio(_arr(o["agent"]["qpos"][0]),
                                 _arr(o["extra"]["goal_pos"][0]) if use_goal else None)
            while len(rgb_hist) < obs_horizon:
                rgb_hist.append(rgb); prop_hist.append(prop)
            rgb_hist.append(rgb); prop_hist.append(prop)

        push(obs)
        st = dict(grasped=False, placed=False, success=False, steps=0,
                  dmin=float(np.linalg.norm(_p3(u.agent.tcp.pose.p) - _p3(u.cube.pose.p))))
        t = 0
        while t < max_steps and not st["success"]:
            rgb = preprocess_obs_rgb(
                torch.stack([torch.as_tensor(f) for f in rgb_hist]), img_size
            ).unsqueeze(0).to(device)
            proprio = torch.stack([torch.as_tensor(p) for p in prop_hist]
                                  ).unsqueeze(0).float().to(device)
            chunk = policy.predict_action(rgb, proprio, [instruction],
                                          n_steps=n_steps, guidance=guidance)[0]
            for k in range(min(H, chunk.shape[0])):
                obs, _, term, _, info = env.step(chunk[k].cpu().numpy())
                push(obs); t += 1
                st["dmin"] = min(st["dmin"], float(np.linalg.norm(
                    _p3(u.agent.tcp.pose.p) - _p3(u.cube.pose.p))))
                # 这三个标志是环境自己的成功判据拆开的，不是我们定的阈值
                st["grasped"] |= _b(info["is_grasped"])
                st["placed"] |= _b(info["is_obj_placed"])
                st["success"] = _b(info["success"])
                if st["success"] or _b(term) or t >= max_steps:
                    break
        st["steps"] = t
        st["stage"] = classify(st)
        rows.append(st)
    return rows


def summarize(rows):
    counts = {s: 0 for s in STAGES}
    for r in rows:
        counts[r["stage"]] += 1
    return {s: c / len(rows) for s, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="采样步数；默认用 checkpoint 里该模型的配置")
    ap.add_argument("--ema", action="store_true",
                    help="用 EMA 权重。默认在线权重 —— 本项目里 EMA 读数是坏的，见 docs/next-steps.md")
    ap.add_argument("--out-json", type=Path, default=Path("outputs/failure_modes.json"))
    ap.add_argument("--fig", type=Path, default=Path("docs/figures/failure_modes.png"))
    args = ap.parse_args()

    labels = args.labels or [c.parent.name for c in args.ckpt]
    assert len(labels) == len(args.ckpt), "--labels 数量要和 --ckpt 对齐"

    results = {}
    for label, ckpt in zip(labels, args.ckpt):
        policy, cfg, name = load_policy(ckpt, use_ema=args.ema)
        steps = args.n_steps or cfg["model"].get("n_sample_steps", 10)
        env = make_env(args.task)
        try:
            rows = run(policy, env, args.task, cfg,
                       n_episodes=args.n_episodes, n_steps=steps)
        finally:
            env.close()
        results[label] = summarize(rows)
        print(f"\n===== {label}  ({name}, {steps} 步, n={args.n_episodes}) =====")
        for s in STAGES:
            frac = results[label][s]
            print(f"  {s:<12} {100*frac:5.1f} %  {'█' * int(round(frac * 40))}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    plot(results, args.fig, args.task, args.n_episodes)
    print(f"\n已写出 {args.out_json} 与 {args.fig}")


def plot(results, path: Path, task: str, n_episodes: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 4.5, 4.4))
    bottom = np.zeros(len(labels))
    for stage, stage_en, color in zip(STAGES, STAGES_EN, STAGE_COLORS):
        vals = np.array([100 * results[l][stage] for l in labels])
        ax.bar(labels, vals, bottom=bottom, color=color, label=stage_en,
               edgecolor="white", linewidth=0.8, width=0.6)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        bottom += vals

    ax.set_ylabel("share of episodes (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Where the policy stops: failure attribution "
                 f"({task}, n={n_episodes})")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
