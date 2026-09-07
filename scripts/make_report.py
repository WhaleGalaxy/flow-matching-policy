"""把 results.csv 汇总成 README 用的结果表和图。

    python scripts/make_report.py --csv outputs/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SHORT = {"PickCube-v1": "PickCube", "StackCube-v1": "StackCube",
         "PegInsertionSide-v1": "PegInsertion"}
LABEL = {"bc": "BC", "ddpm": "Diffusion Policy", "fm": "FM (Ours)"}
COLOR = {"bc": "#8B8B8B", "ddpm": "#C4682B", "fm": "#378ADD"}


def agg(df):
    """按 (模型, 任务) 聚合多个 seed，给出 mean±std。"""
    g = df.groupby(["model", "task"])["success_rate"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def main_table(df: pd.DataFrame) -> str:
    """主结果表：各方法 × 各任务的成功率。"""
    base = df[(df.n_steps.isin([10, 100])) & (df.guidance == 1.0)]
    a = agg(base)
    tasks = [t for t in SHORT if t in set(a.task)]
    lines = ["| Method | " + " | ".join(SHORT[t] for t in tasks) + " |",
             "|---|" + "---|" * len(tasks)]
    for m in ("bc", "ddpm", "fm"):
        sub = a[a.model == m]
        if sub.empty:
            continue
        cells = []
        for t in tasks:
            r = sub[sub.task == t]
            cells.append(f"{100*r['mean'].iloc[0]:.0f}±{100*r['std'].iloc[0]:.0f}%"
                         if not r.empty else "—")
        name = f"**{LABEL[m]}**" if m == "fm" else LABEL[m]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_all(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    # --- 图1：主结果分组柱状图 ---
    base = df[(df.n_steps.isin([10, 100])) & (df.guidance == 1.0)]
    a = agg(base)
    tasks = [t for t in SHORT if t in set(a.task)]
    models = [m for m in ("bc", "ddpm", "fm") if m in set(a.model)]
    if tasks and models:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            sub = a[a.model == m].set_index("task")
            xs = [j + i * w - 0.4 + w / 2 for j in range(len(tasks))]
            ys = [100 * sub.loc[t, "mean"] if t in sub.index else 0 for t in tasks]
            es = [100 * sub.loc[t, "std"] if t in sub.index else 0 for t in tasks]
            ax.bar(xs, ys, w * 0.9, yerr=es, capsize=3, label=LABEL[m], color=COLOR[m])
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([SHORT[t] for t in tasks])
        ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.set_title("ManiSkill3 success rate (100 episodes)")
        plt.tight_layout(); p = out_dir / "results_main.png"
        plt.savefig(p, dpi=140); figs.append(p); plt.close()

    # --- 图2：采样步数 vs 成功率 ---
    sw = df[(df.guidance == 1.0)].groupby(["model", "task", "n_steps"])["success_rate"].mean().reset_index()
    if sw.n_steps.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, t), grp in sw.groupby(["model", "task"]):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("n_steps")
            ax.plot(grp.n_steps, 100 * grp.success_rate, "o-",
                    color=COLOR.get(m, "#666"),
                    ls="--" if m == "ddpm" else "-",
                    label=f"{LABEL.get(m,m)} · {SHORT.get(t,t)}")
        ax.set_xscale("log"); ax.set_xlabel("Sampling steps")
        ax.set_ylabel("Success rate (%)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8); ax.set_title("Quality vs. sampling steps")
        plt.tight_layout(); p = out_dir / "results_steps.png"
        plt.savefig(p, dpi=140); figs.append(p); plt.close()

    # --- 图3：CFG 权重消融 ---
    gw = df[df.guidance.notna()].groupby(["task", "guidance"])["success_rate"].mean().reset_index()
    if gw.guidance.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for t, grp in gw.groupby("task"):
            grp = grp.sort_values("guidance")
            ax.plot(grp.guidance, 100 * grp.success_rate, "o-", label=SHORT.get(t, t))
        ax.set_xlabel("CFG guidance weight w"); ax.set_ylabel("Success rate (%)")
        ax.grid(alpha=0.3); ax.legend(); ax.set_title("Classifier-free guidance ablation")
        plt.tight_layout(); p = out_dir / "results_cfg.png"
        plt.savefig(p, dpi=140); figs.append(p); plt.close()

    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["success_rate"] = df.success_rate.astype(float)
    print(f"{len(df)} 条评测记录\n")
    print("主结果表：\n")
    print(main_table(df))
    figs = plot_all(df, args.out_dir)
    print("\n生成的图：")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
