"""连续指标诊断：策略到底有没有在朝方块走。

成功率在低分区间几乎没有分辨率——25 个 episode 下 1 次和 2 次成功的置信区间
几乎完全重叠，看不出趋势。这个脚本记录 rollout 里的连续量，几个 episode
就能区分"学到了接近但抓不稳"和"根本没在朝目标走"（见 docs/debugging.md）。

    python scripts/diagnose_rollout.py --ckpt outputs/fm_PickCube_s42/ckpt_8000.pt
    python scripts/diagnose_rollout.py --demo          # 演示数据的参考值
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.evaluate import make_env, obs_kwargs, task_goal_pos  # noqa: E402
from src.data.dataset import (TASK_INSTRUCTIONS, build_proprio,  # noqa: E402
                              preprocess_obs_rgb)
from run_eval import load_policy  # noqa: E402


def _np(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _p(x):
    return np.asarray(x.cpu() if torch.is_tensor(x) else x).reshape(-1)[:3]


@torch.no_grad()
def manip_obj(unwrapped):
    """被操作的物体在不同任务里叫不同的名字。写死 `u.cube` 会在 StackCube 上
    崩成 `'StackCubeEnv' object has no attribute 'cube'`。"""
    for name in ("cube", "cubeA", "obj", "peg"):
        if hasattr(unwrapped, name):
            return getattr(unwrapped, name)
    raise AttributeError(f"{type(unwrapped).__name__} 上找不到被操作物体")


def diagnose(policy, env, task, n_episodes=10, obs_horizon=2, execute_horizon=8,
             img_size=126, n_steps=10, guidance=1.0, max_steps=300, seed=0,
             use_goal=False, goal_slot=False, task_goal=False):
    from collections import deque
    device = next(policy.parameters()).device
    instruction = TASK_INSTRUCTIONS[task]
    u = env.unwrapped
    rows = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        rgb_hist, prop_hist = deque(maxlen=obs_horizon), deque(maxlen=obs_horizon)

        def push(o):
            # proprio 的组装必须与训练完全一致，走 dataset 里同一个 build_proprio。
            # 这里曾经直接用 qpos，遇到 use_goal 的 checkpoint 会撞出
            # `mat1 and mat2 shapes cannot be multiplied (1x9 and 12x128)`。
            rgb = o["sensor_data"]["base_camera"]["rgb"][0]
            if task_goal:
                goal = task_goal_pos(env, task)
            elif use_goal or (goal_slot and "goal_pos" in o["extra"]):
                goal = _np(o["extra"]["goal_pos"][0])
            else:
                goal = None
            prop = build_proprio(_np(o["agent"]["qpos"][0]), goal, goal_slot=goal_slot)
            while len(rgb_hist) < obs_horizon:
                rgb_hist.append(rgb); prop_hist.append(prop)
            rgb_hist.append(rgb); prop_hist.append(prop)

        push(obs)
        d_tcp, cube_z, grasped, success, t = [], [], False, False, 0
        d_tcp.append(np.linalg.norm(_p(u.agent.tcp.pose.p) - _p(manip_obj(u).pose.p)))
        z0 = _p(manip_obj(u).pose.p)[2]

        while t < max_steps and not success:
            rgb = preprocess_obs_rgb(
                torch.stack([torch.as_tensor(f) for f in rgb_hist]), img_size
            ).unsqueeze(0).to(device)
            proprio = torch.stack([torch.as_tensor(p) for p in prop_hist]
                                  ).unsqueeze(0).float().to(device)
            chunk = policy.predict_action(rgb, proprio, [instruction],
                                          n_steps=n_steps, guidance=guidance)[0]
            for k in range(min(execute_horizon, chunk.shape[0])):
                obs, _, term, trunc, info = env.step(chunk[k].cpu().numpy())
                push(obs); t += 1
                d_tcp.append(np.linalg.norm(_p(u.agent.tcp.pose.p) - _p(manip_obj(u).pose.p)))
                cube_z.append(_p(manip_obj(u).pose.p)[2])
                grasped = grasped or bool(np.asarray(u.agent.is_grasping(manip_obj(u)).cpu()).reshape(-1)[0])
                success = bool(np.asarray(info["success"]).reshape(-1)[0]) if not torch.is_tensor(info["success"]) \
                    else bool(info["success"].cpu().reshape(-1)[0])
                if success or bool(np.asarray(term.cpu() if torch.is_tensor(term) else term).reshape(-1)[0]) \
                   or t >= max_steps:
                    break

        rows.append(dict(ep=ep, d0=d_tcp[0], dmin=min(d_tcp), dend=d_tcp[-1],
                         lift=max(cube_z) - z0 if cube_z else 0.0,
                         grasped=grasped, success=success, steps=t))
    return rows


def report(rows, title):
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in
         ("d0", "dmin", "dend", "lift", "grasped", "success", "steps")}
    print(f"\n===== {title}  (n={len(rows)}) =====")
    print(f"  起始 TCP→方块距离   {a['d0'].mean():.3f} m")
    print(f"  最近 TCP→方块距离   {a['dmin'].mean():.3f} m   "
          f"(min {a['dmin'].min():.3f} / max {a['dmin'].max():.3f})")
    print(f"  结束 TCP→方块距离   {a['dend'].mean():.3f} m")
    print(f"  方块最大抬升        {a['lift'].mean()*1000:.1f} mm")
    print(f"  抓到过方块          {a['grasped'].mean()*100:.0f} %")
    print(f"  成功                {a['success'].mean()*100:.0f} %")
    print(f"\n  判读：dmin 明显小于 d0 → 策略在朝方块走；dmin ≈ d0 或更大 → 没学到。")
    print(f"        方块边长 20mm，抓取需要 dmin ≲ 0.02-0.03 m。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--no-ema", action="store_true",
                    help="用在线权重而非 EMA（EMA 权重缺失的旧 checkpoint 需要）")
    args = ap.parse_args()

    policy, cfg, name = load_policy(args.ckpt, use_ema=not args.no_ema)
    env = make_env(args.task)
    try:
        rows = diagnose(policy, env, args.task, n_episodes=args.n_episodes,
                        obs_horizon=cfg["obs_horizon"], img_size=cfg["img_size"],
                        execute_horizon=cfg["eval"]["execute_horizon"],
                        n_steps=args.n_steps, guidance=args.guidance,
                        **obs_kwargs(cfg))
    finally:
        env.close()
    report(rows, f"{name} @ {args.ckpt.name}  {args.task}")


if __name__ == "__main__":
    main()
