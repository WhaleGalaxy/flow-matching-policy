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


def _np(x):
    """ManiSkill3 返回的观测/渲染结果是 CUDA tensor，转 numpy 前需先回 CPU。"""
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import (CONTROL_MODE, TASK_GOAL_ACTOR,  # noqa: E402
                              TASK_INSTRUCTIONS, build_proprio, preprocess_obs_rgb)


def obs_kwargs(cfg) -> dict:
    """从 checkpoint 的 cfg 里取出**全部**观测配置，一次性传给 rollout。

    存在的理由：观测配置现在有四项（use_goal / goal_slot / task_goal / use_proprio），
    每个评测脚本各自手工挑几项传，加一项就断一处 —— 本项目已经因此断过三次
    （diagnose_rollout 两次、language_ablation 一次），症状都是
    `mat1 and mat2 shapes cannot be multiplied (1x9 and 12x128)`。
    只要新配置项加进这个函数，所有调用方自动跟上。
    """
    return dict(use_goal=cfg.get("use_goal", False),
                goal_slot=cfg.get("goal_slot", False),
                task_goal=cfg.get("task_goal", False))


def task_goal_pos(env, task: str) -> np.ndarray:
    """从**运行中的环境**读这个任务的目标位置。

    训练侧读的是演示 h5 里的 `env_states/actors/<name>` 前 3 维，两者必须指向
    同一个量，否则训练和评测的第 10-12 维含义不同 —— 而这种错位是静默的：
    维度对得上，loss 正常，策略只是学不会送达。已逐位对拍验证一致。
    """
    actor = TASK_GOAL_ACTOR.get(task)
    assert actor, f"{task} 没有登记目标 actor，见 src/data/dataset.TASK_GOAL_ACTOR"
    p = getattr(env.unwrapped, actor).pose.p
    return _np(p).reshape(-1)[:3]


# 评测环境的控制模式**必须**与训练数据一致，且必须显式指定 ——
# gym.make 的默认是 pd_joint_delta_pos，与我们的数据不符时，策略输出会被
# 错误解释并裁剪，成功率会锁死在一个与训练进度无关的低值上（见 docs/debugging.md）。
def make_env(task: str, num_envs: int = 1, control_mode: str = CONTROL_MODE):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        task, obs_mode="rgb", render_mode="rgb_array", control_mode=control_mode,
        num_envs=num_envs, sim_backend="physx_cpu" if num_envs == 1 else "physx_cuda",
    )
    assert env.unwrapped.control_mode == control_mode, (
        f"控制模式不符：环境用 {env.unwrapped.control_mode}，演示数据是 {control_mode}")
    return env


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
    use_goal: bool = False,
    goal_slot: bool = False,
    task_goal: bool = False,
    n_average: int = 1,
    instruction: str | None = None,
) -> dict:
    """`instruction` 传入时覆盖该任务的默认指令。

    这是"语言到底有没有被用上"那个消融的入口：把别的任务的指令、或者 CFG 用的
    空串喂进去，成功率掉多少就是语言在这个策略里实际承载了多少信息。掉不动的话，
    说明任务是从像素里读出来的，"语言条件"这四个字就不能写进结论。
    """
    device = next(policy.parameters()).device
    instruction = TASK_INSTRUCTIONS[task] if instruction is None else instruction
    policy.eval()

    successes, lengths, frames = [], [], []
    for ep in range(n_episodes):
        # 固定采样噪声。这消除了一个方差来源，但**评测仍然不是 bit 级可复现的**：
        # 注意力的 SDPA 后端按启发式在 flash / mem-efficient 之间切换，带来 1e-6
        # 量级的数值差异，而 rollout 是闭环 —— 动作末位的差异改变仿真状态，
        # 几百步后轨迹完全不同。实测同一 checkpoint 重测的波动约 1 个百分点
        # （n=100），大于本项目各方法之间的差距。
        #
        # 结论不是去追求 bit 级确定性（要 use_deterministic_algorithms 且明显变慢），
        # 而是：成功率在个位数区间不足以支撑结论，用 diagnose_rollout.py 的连续指标。
        torch.manual_seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        # 观测窗口：起始时用第一帧重复填满
        rgb_hist = deque(maxlen=obs_horizon)
        prop_hist = deque(maxlen=obs_horizon)

        def push(o):
            rgb = o["sensor_data"][camera]["rgb"][0]        # (128,128,3) uint8
            # proprio 的组装必须与训练完全一致，走同一个 build_proprio
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
        success, t = False, 0
        while t < max_steps and not success:
            rgb = preprocess_obs_rgb(
                torch.stack([torch.as_tensor(f) for f in rgb_hist]), img_size
            ).unsqueeze(0).to(device)
            proprio = torch.stack(
                [torch.as_tensor(p) for p in prop_hist]).unsqueeze(0).float().to(device)

            chunk = policy.predict_action(
                rgb, proprio, [instruction], n_steps=n_steps, guidance=guidance,
                n_average=n_average)[0]

            for k in range(min(execute_horizon, chunk.shape[0])):
                obs, _, terminated, truncated, info = env.step(chunk[k].cpu().numpy())
                if record_frames and ep == 0:
                    frames.append(_np(env.render()[0]))
                push(obs)
                t += 1
                success = bool(_np(info["success"]).reshape(-1)[0])
                if success or bool(_np(terminated).reshape(-1)[0]) or t >= max_steps:
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
