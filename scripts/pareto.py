"""主图：成功率 vs 机器人实际能跑的闭环控制频率。

单看"FM 10 步比 DDPM 100 步快 9.3 倍"是算术不是结果 —— 同一个骨干跑十分之一的
步数，快十倍是定义决定的。有意义的问题是**少步的 FM 和少步的 DDPM 谁的成功率
掉得更快**，以及换算到部署上差多少。

横轴的推导：控制回路跑 f Hz，一次推理执行 H 步，那么必须在 H/f 秒内算完下一块，
否则机器人会在等下一个动作块时空转。所以可持续的最高控制频率是

    f_max = execute_horizon / 推理延迟

**这个式子假设推理与执行是流水的** —— 在执行第 k 块的同时算第 k+1 块，
所以只要延迟不超过一块的执行时长 H/f，机器人就不会空等。真实系统都这么做。
若是同步实现（算的时候机器人停住），达到的平均频率是 H/(H/f + 延迟)，
比这里给的上界低。另有一个动作分块本身固有的代价：流水意味着执行中的动作块
是用一块之前的观测算出来的，这部分滞后不体现在这张图里。

于是每个 (方法, 采样步数, 执行长度) 组合给出一个 (f_max, 成功率) 点。
取左上包络 —— "要求至少 f Hz 时，这个方法最好能做到多少成功率" —— 就是它的
部署可行域。这才是机器人岗位真正会问的问题。

延迟在本机实测，成功率从 run_eval 的 sweep CSV 读。

    python scripts/run_eval.py --ckpt <fm> --no-ema --sweep-steps 1 2 4 8 16 \
        --sweep-horizon 1 2 4 8 16 --out outputs/sweeps.csv
    python scripts/pareto.py --csv outputs/sweeps.csv \
        --ckpt fm=<fm.pt> ddpm=<ddpm.pt> bc=<bc.pt>
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_eval import load_policy  # noqa: E402

STYLE = {
    "fm":   dict(color="#2f6f4e", marker="o", label="Flow Matching"),
    "ddpm": dict(color="#8c2d1e", marker="s", label="Diffusion (DDPM/DDIM)"),
    "bc":   dict(color="#6b6b6b", marker="^", label="Behaviour cloning"),
}


@torch.no_grad()
def measure_latency(policy, n_steps: int, guidance: float = 1.0,
                    iters: int = 30, warmup: int = 8) -> float:
    """单次动作块预测的延迟（ms），batch=1 —— 部署时面对的是一个机器人。

    用真实 checkpoint 而不是随机初始化：延迟本身与权重无关，但这样能保证
    测的是评测里跑的那个结构（proprio 维度、CFG 与否都会影响）。
    """
    device = next(policy.parameters()).device
    cfg_dim = policy.encoder.proprio.net[0].in_features if policy.encoder.use_proprio else 9
    rgb = torch.randn(1, 2, 3, policy.encoder.visual.img_size,
                      policy.encoder.visual.img_size, device=device)
    prop = torch.randn(1, 2, cfg_dim, device=device)
    inst = ["Pick up the red cube"]

    for _ in range(warmup):
        policy.predict_action(rgb, prop, inst, n_steps=n_steps, guidance=guidance)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        policy.predict_action(rgb, prop, inst, n_steps=n_steps, guidance=guidance)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def read_success(csv_path: Path, task: str, ema: bool = False) -> dict:
    """(model, n_steps, execute_horizon) -> 成功率。同一配置多行时取平均。"""
    acc = defaultdict(list)
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["task"] != task or (r["ema"] == "True") != ema:
                continue
            if float(r["guidance"]) != 1.0:
                continue          # CFG 是另一条消融，不混进这张图
            key = (r["model"], int(r["n_steps"]), int(r["execute_horizon"]))
            acc[key].append(float(r["success_rate"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def envelope(points):
    """左上包络：要求至少 f Hz 时，最好能拿到多少成功率。

    按频率从高到低扫，取成功率的running max —— 一个更慢又更差的配置没有存在意义。
    """
    pts = sorted(points, key=lambda p: -p[0])
    out, best = [], -1.0
    for f, sr, meta in pts:
        if sr > best:
            best = sr
            out.append((f, sr, meta))
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="model=path 形式，例如 fm=outputs/fmgoal.../ckpt_20000.pt")
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--fig", type=Path, default=Path("docs/figures/pareto_control_rate.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("outputs/pareto_points.csv"))
    args = ap.parse_args()

    ckpts = dict(kv.split("=", 1) for kv in args.ckpt)
    success = read_success(args.csv, args.task)
    assert success, f"{args.csv} 里没有 {args.task} 的在线权重结果"

    # 每个 (model, n_steps) 只需测一次延迟，与 execute_horizon 无关
    need = sorted({(m, s) for (m, s, _) in success if m in ckpts})
    latency: dict[tuple[str, int], float] = {}
    for model in sorted({m for m, _ in need}):
        policy, _, _ = load_policy(Path(ckpts[model]), use_ema=False)
        for m, s in need:
            if m == model:
                latency[(m, s)] = measure_latency(policy, s)
                print(f"  延迟 {m:<5} {s:>3} 步 : {latency[(m, s)]:6.2f} ms", flush=True)
        del policy
        torch.cuda.empty_cache()

    rows, by_model = [], defaultdict(list)
    for (model, steps, H), sr in sorted(success.items()):
        if (model, steps) not in latency:
            continue
        f_max = H / (latency[(model, steps)] / 1000.0)
        rows.append(dict(model=model, n_steps=steps, execute_horizon=H,
                         latency_ms=round(latency[(model, steps)], 2),
                         max_control_hz=round(f_max, 1), success_rate=sr))
        by_model[model].append((f_max, sr, (steps, H)))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'模型':<6}{'采样步':>7}{'执行长度':>9}{'延迟ms':>9}{'可达Hz':>9}{'成功率':>9}")
    print("-" * 50)
    for r in sorted(rows, key=lambda r: (r["model"], -r["max_control_hz"])):
        print(f"{r['model']:<6}{r['n_steps']:>7}{r['execute_horizon']:>9}"
              f"{r['latency_ms']:>9.2f}{r['max_control_hz']:>9.0f}"
              f"{100*r['success_rate']:>8.0f}%")

    plot(by_model, args.fig, args.task)
    print(f"\n已写出 {args.out_csv} 与 {args.fig}")


def plot(by_model, path: Path, task: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for model, pts in by_model.items():
        st = STYLE.get(model, dict(color="k", marker="x", label=model))
        f = np.array([p[0] for p in pts]); sr = 100 * np.array([p[1] for p in pts])
        ax.scatter(f, sr, s=26, alpha=0.35, color=st["color"], marker=st["marker"],
                   linewidths=0)
        env = envelope(pts)
        ax.plot([p[0] for p in env], [100 * p[1] for p in env], drawstyle="steps-post",
                color=st["color"], lw=2.2, marker=st["marker"], ms=6, label=st["label"])
        # 只标包络上的点，标全部会糊成一片。BC 是一次前向，没有采样步数可言。
        for fx, sy, (steps, H) in env:
            tag = f"H={H}" if model == "bc" else f"{steps} steps / H={H}"
            ax.annotate(tag, (fx, 100 * sy), textcoords="offset points",
                        xytext=(4, 5), fontsize=7.5, color=st["color"])

    ax.set_xscale("log")
    ax.set_xlabel("Sustainable closed-loop control rate   "
                  r"$f_{max}$ = execute_horizon / inference latency   (Hz, log)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Deployment frontier: success rate reachable at a given "
                 f"control rate ({task})")
    ax.grid(alpha=0.25, which="both", lw=0.6)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
