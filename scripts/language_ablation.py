"""语言到底有没有被用上？

多任务策略跑通了不等于"语言条件"成立。如果三个任务的场景一眼就能分辨，
策略完全可以从像素里读出该做什么，指令那一路白接了 —— 而 README 里
"语言条件的多任务策略"这句话就没有证据。

判据很直接：**评测时把指令换掉，看成功率掉多少。**

    行 = 实际运行的环境，列 = 喂给策略的指令

对角线是正确指令。若离对角线的格子和对角线一样高，说明指令没有承载信息；
若换成别的任务的指令后成功率塌掉，说明策略确实在读语言。null 列（空串）
是 CFG 训练时用的无条件输入，它衡量的是"完全不给语言"时还剩多少。

这个实验不需要重新训练，只改评测时喂进去的字符串。

    python scripts/language_ablation.py --ckpt outputs/fmmt_.../ckpt_20000.pt --n-episodes 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import load_policy  # noqa: E402
from src.data.dataset import TASK_INSTRUCTIONS  # noqa: E402
from src.evaluate import make_env, obs_kwargs, rollout  # noqa: E402

NULL = ""   # CFG 的 null condition，与训练时用的是同一个


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=None, help="默认用 checkpoint 里的任务列表")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--out-json", type=Path, default=Path("outputs/language_ablation.json"))
    ap.add_argument("--fig", type=Path, default=Path("docs/figures/language_ablation.png"))
    args = ap.parse_args()

    policy, cfg, name = load_policy(args.ckpt, use_ema=args.ema)
    tasks = args.tasks or cfg["tasks"]
    steps = args.n_steps or cfg["model"].get("n_sample_steps", 1 if name.startswith("bc") else 10)
    # 列：每个任务的正确指令，外加一个 null
    cols = [TASK_INSTRUCTIONS[t] for t in tasks] + [NULL]
    col_labels = [t.split("-")[0] for t in tasks] + ["null"]

    print(f"checkpoint: {args.ckpt}  模型={name}  采样步数={steps}")
    print(f"任务={tasks}  每格 {args.n_episodes} episode\n", flush=True)

    grid = {}
    for task in tasks:
        env = make_env(task)
        try:
            for lab, instr in zip(col_labels, cols):
                t0 = time.perf_counter()
                r = rollout(policy, env, task, n_episodes=args.n_episodes,
                            obs_horizon=cfg["obs_horizon"],
                            execute_horizon=cfg["eval"]["execute_horizon"],
                            img_size=cfg["img_size"], n_steps=steps, guidance=1.0,
                            instruction=instr, **obs_kwargs(cfg))
                grid[(task, lab)] = r["success_rate"]
                mark = " ←正确" if lab == task.split("-")[0] else ""
                print(f"  环境 {task:16s} 指令 {lab:12s} SR {100*r['success_rate']:5.1f}%"
                      f"  ({time.perf_counter()-t0:.0f}s){mark}", flush=True)
        finally:
            env.close()

    blob = {"ckpt": args.ckpt.as_posix(), "model": name, "tasks": tasks,
            "cols": col_labels, "n_episodes": args.n_episodes,
            "grid": {f"{t}|{c}": v for (t, c), v in grid.items()}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(blob, indent=2))

    # 一个可以直接读的数字：正确指令 vs 错误指令的平均差
    diag = [grid[(t, t.split("-")[0])] for t in tasks]
    off = [v for (t, c), v in grid.items() if c not in ("null", t.split("-")[0])]
    nulls = [grid[(t, "null")] for t in tasks]
    print(f"\n正确指令 {100*sum(diag)/len(diag):.1f}%   "
          f"错误指令 {100*sum(off)/len(off):.1f}%   "
          f"无指令 {100*sum(nulls)/len(nulls):.1f}%")
    print("三者接近就说明语言没有被用上，任务是从像素里读出来的。")

    plot(tasks, col_labels, grid, args.fig)
    print(f"已写出 {args.out_json} 与 {args.fig}")


def plot(tasks, col_labels, grid, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M = np.array([[grid[(t, c)] for c in col_labels] for t in tasks]) * 100
    fig, ax = plt.subplots(figsize=(1.35 * len(col_labels) + 2.4, 1.0 * len(tasks) + 2.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(100 * 0.01, M.max()))
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)), [t.split("-")[0] for t in tasks])
    ax.set_xlabel("instruction fed to the policy")
    ax.set_ylabel("environment actually run")
    ax.set_title("Is the policy actually reading the instruction?\n"
                 "off-diagonal = wrong task's instruction, null = empty string", fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] < M.max() * 0.6 else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="success rate (%)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)


if __name__ == "__main__":
    main()
