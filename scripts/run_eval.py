"""从 checkpoint 做完整评测，输出 CSV 供汇总成结果表。

    python scripts/run_eval.py --ckpt outputs/fm_PickCube_s42/ckpt_20000.pt \
        --tasks PickCube-v1 --n-episodes 100

    # 采样步数消融（同一个 checkpoint，不用重训）
    python scripts/run_eval.py --ckpt ... --sweep-steps 1 2 5 10 20

    # CFG 权重消融
    python scripts/run_eval.py --ckpt ... --sweep-guidance 1.0 1.5 2.0
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_env, rollout  # noqa: E402
from src.models.policy import (BCPolicy, DDPMPolicy, FMPolicy,  # noqa: E402
                               load_trainable_state_dict)

BUILDERS = {"fm": FMPolicy, "bc": BCPolicy, "ddpm": DDPMPolicy}


def load_policy(ckpt_path: Path, use_ema: bool = True, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    name = cfg["model"]["name"]
    common = dict(act_dim=8, act_horizon=cfg["act_horizon"], img_size=cfg["img_size"],
                  proprio_dim=9, use_proprio=cfg["use_proprio"],
                  d_model=cfg["model"]["d_model"])
    if name == "bc":
        policy = BCPolicy(**common)
    else:
        gen = dict(n_layers=cfg["model"]["n_layers"], n_heads=cfg["model"]["n_heads"],
                   cfg_dropout=cfg["model"].get("cfg_dropout", 0.2), **common)
        policy = (DDPMPolicy(n_train_steps=cfg["model"]["n_train_steps"], **gen)
                  if name == "ddpm" else FMPolicy(**gen))

    state = ck["ema" if use_ema else "model"]
    # 兼容早期保存的完整 state_dict
    try:
        load_trainable_state_dict(policy, state)
    except AssertionError:
        policy.load_state_dict(state)
    assert bool(policy.normalizer.fitted), "checkpoint 里的归一化统计量没拟合过"
    return policy.to(device).eval(), cfg, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--sweep-steps", type=int, nargs="+", default=None)
    ap.add_argument("--sweep-guidance", type=float, nargs="+", default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/results.csv"))
    args = ap.parse_args()

    policy, cfg, name = load_policy(args.ckpt, use_ema=not args.no_ema)
    tasks = args.tasks or cfg["tasks"]
    default_steps = cfg["model"].get("n_sample_steps", 10)
    steps_list = args.sweep_steps or [default_steps]
    guid_list = args.sweep_guidance or [1.0]

    print(f"checkpoint: {args.ckpt}  模型={name}  EMA={not args.no_ema}")
    print(f"任务={tasks}  episodes={args.n_episodes}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["ckpt", "model", "seed", "task", "n_steps", "guidance",
                        "ema", "n_episodes", "success_rate", "mean_length"])

        for task in tasks:
            env = make_env(task)
            try:
                for s in steps_list:
                    for g in guid_list:
                        t0 = time.perf_counter()
                        r = rollout(policy, env, task, n_episodes=args.n_episodes,
                                    obs_horizon=cfg["obs_horizon"],
                                    execute_horizon=cfg["eval"]["execute_horizon"],
                                    img_size=cfg["img_size"], n_steps=s, guidance=g)
                        print(f"  {task:22s} steps={s:<4} w={g:<4} "
                              f"SR {100*r['success_rate']:5.1f}%  "
                              f"步长 {r['mean_length']:5.1f}  "
                              f"({time.perf_counter()-t0:.0f}s)", flush=True)
                        w.writerow([args.ckpt.as_posix(), name, cfg["seed"], task, s, g,
                                    not args.no_ema, args.n_episodes,
                                    f"{r['success_rate']:.4f}", f"{r['mean_length']:.2f}"])
                        fh.flush()
            finally:
                env.close()
    print(f"\n结果已追加到 {args.out}")


if __name__ == "__main__":
    main()
