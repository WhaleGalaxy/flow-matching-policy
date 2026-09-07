"""在 ManiSkill3 环境里 rollout 策略，统计成功率。

动作分块（action chunking）：策略一次预测 act_horizon 步，但只执行前
execute_horizon 步就重新规划。全部执行完再规划会因为开环太久而漂移；
每步都重规划则丢掉了分块带来的时序一致性，也慢 act_horizon 倍。
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TASK_INSTRUCTIONS, preprocess_obs_rgb  # noqa: E402


def make_env(task: str, num_envs: int = 1, render: bool = False):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        task, obs_mode="rgb", render_mode="rgb_array",
        num_envs=num_envs, sim_backend="physx_cpu" if num_envs == 1 else "physx_cuda",
    )


@torch.no_grad()
def rollout(
    policy,
    env,
    task: str,
    n_episodes: int = 100,
    obs_horizon: int = 2,
    execute_horizon: int = 8,
    img_size: int = 126,
    camera: str = "base_camera",
    n_steps: int = 10,
    guidance: float = 1.0,
    max_steps: int = 300,
    seed: int = 0,
    record_frames: bool = False,
) -> dict:
    device = next(policy.parameters()).device
    instruction = TASK_INSTRUCTIONS[task]
    policy.eval()

    successes, lengths, frames = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # 观测窗口：起始时用第一帧重复填满
        rgb_hist = deque(maxlen=obs_horizon)
        prop_hist = deque(maxlen=obs_horizon)

        def push(o):
            rgb = o["sensor_data"][camera]["rgb"][0]        # (128,128,3) uint8
            qpos = o["agent"]["qpos"][0]                     # (9,)
            while len(rgb_hist) < obs_horizon:
                rgb_hist.append(rgb); prop_hist.append(qpos)
            rgb_hist.append(rgb); prop_hist.append(qpos)

        push(obs)
        success, t = False, 0
        while t < max_steps and not success:
            rgb = preprocess_obs_rgb(
                torch.stack([torch.as_tensor(f) for f in rgb_hist]), img_size
            ).unsqueeze(0).to(device)
            proprio = torch.stack(
                [torch.as_tensor(p) for p in prop_hist]).unsqueeze(0).float().to(device)

            chunk = policy.predict_action(
                rgb, proprio, [instruction], n_steps=n_steps, guidance=guidance)[0]

            for k in range(min(execute_horizon, chunk.shape[0])):
                obs, _, terminated, truncated, info = env.step(chunk[k].cpu().numpy())
                if record_frames and ep == 0:
                    frames.append(np.asarray(env.render()[0]))
                push(obs)
                t += 1
                success = bool(np.asarray(info["success"]).reshape(-1)[0])
                if success or bool(np.asarray(terminated).reshape(-1)[0]) or t >= max_steps:
                    break

        successes.append(float(success))
        lengths.append(t)

    out = {
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
    }
    if record_frames:
        out["frames"] = frames
    return out


def evaluate(policy, tasks: list[str], **kw) -> dict:
    """对多个任务分别 rollout，返回 {task: success_rate}。"""
    results = {}
    for task in tasks:
        env = make_env(task)
        try:
            results[task] = rollout(policy, env, task, **kw)
        finally:
            env.close()
    return results
