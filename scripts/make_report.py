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

SHORT = {"PickCube-v1": "PickCube", "PushCube-v1": "PushCube",
         "PullCube-v1": "PullCube", "StackCube-v1": "StackCube",
         "PegInsertionSide-v1": "PegInsertion"}
LABEL = {"bc": "BC · mean-pool", "bc_xattn": "BC · cross-attn", "ddpm": "Diffusion Policy",
         "fm": "FM (Ours)", "fm/multi": "FM (Ours, multi-task)",
         "bc/multi": "BC mean-pool, multi-task", "bc_xattn/multi": "BC cross-attn, multi-task",
         "ddpm/multi": "Diffusion Policy, multi-task",
         "bc/nogoal": "BC · no goal_pos", "ddpm/nogoal": "Diffusion Policy · no goal_pos",
         "fm/nogoal": "FM · no goal_pos", "fm/multi/nogoal": "FM multi-task · no goal_pos"}
COLOR = {"bc": "#8B8B8B", "bc_xattn": "#5E5E5E", "ddpm": "#C4682B", "fm": "#378ADD",
         "fm/multi": "#1F5C99", "bc/multi": "#A9A9A9", "bc_xattn/multi": "#6F6F6F",
         "ddpm/multi": "#A9552B",
         "bc/nogoal": "#C9C9C9", "ddpm/nogoal": "#E0AE8E",
         "fm/nogoal": "#A8CBEE", "fm/multi/nogoal": "#8FAECB"}
ORDER = ("bc", "bc_xattn", "ddpm", "fm",
         "bc/multi", "bc_xattn/multi", "ddpm/multi", "fm/multi")

# 每个方法的"默认工作点"：主结果表只取这一行，否则消融扫出来的其它步数会被
# 一起平均进去。BC 是一次前向，DDPM 的常规设置是 100 步，FM 是 10 步。
DEFAULT_STEPS = {"bc": 1, "bc_xattn": 1, "ddpm": 100, "fm": 10}
DEFAULT_HORIZON = 8


def main_rows(df: pd.DataFrame) -> pd.DataFrame:
    """主结果表用的行：各方法在自己的默认工作点、默认执行长度、无 CFG。"""
    # BC 没有采样步数这个旋钮，早期 CSV 里它继承了一个无意义的 10。
    # 对 bc 不筛步数，否则它会整行掉出主表。
    steps_ok = (df.model.map(DEFAULT_STEPS).eq(df.n_steps)
                | df.model.isin(("bc", "bc_xattn")))
    return df[steps_ok & df.execute_horizon.eq(DEFAULT_HORIZON) & df.guidance.eq(1.0)]


def order_of(methods) -> list:
    """ORDER 里的先排前面，其余（如 --weights both 产生的 /online 变体）按名字跟后。"""
    known = [m for m in ORDER if m in methods]
    return known + sorted(m for m in methods if m not in ORDER)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """加上区分实验条件的列，并去掉重复评测。

    两件必须先做的事，否则主结果表是错的：

    1. 消融是复用同一个 checkpoint 跑的，`--sweep-steps 1 2 5 10 20` 里 n_steps=10
       那行、`--sweep-guidance 1.0 1.5 2.0` 里 w=1.0 那行，与主评测行的字段完全
       相同。不去重的话它们会被 groupby 当成额外的 seed，±std 就不再是种子间方差。
    2. 单任务 FM 和多任务 FM 的 model 都是 "fm"、task 都含 PickCube-v1，只有
       checkpoint 路径不同。不区分的话主表里 FM 的 PickCube 一格会把两个不同的
       实验条件平均掉。
    """
    df = df.copy()
    df["ema"] = df.ema.astype(str).str.lower().isin(("true", "1"))
    # 旧 CSV 没有这两列。当时所有评测都跑在默认执行长度下；use_goal 从 run 名字推。
    if "execute_horizon" not in df:
        df["execute_horizon"] = DEFAULT_HORIZON
    if "use_goal" not in df:
        df["use_goal"] = df.ckpt.map(lambda c: "goal" in Path(c).parent.name)
    df["use_goal"] = df.use_goal.astype(str).str.lower().isin(("true", "1"))
    if "goal_slot" not in df:
        df["goal_slot"] = False
    df["goal_slot"] = df.goal_slot.astype(str).str.lower().isin(("true", "1"))
    df["run"] = df.ckpt.map(lambda c: Path(c).parent.name)
    # 多任务 run 的命名：早期是 "fm_PickCube+StackCube+..."，现在是 "fm_mt_s42"
    df["multitask"] = df.run.str.contains(r"\+", regex=True) | df.run.str.contains("_mt")
    df["method"] = df.model.where(~df.multitask, df.model + "/multi")
    # 观测配置进 method，否则 (方法,任务) 聚合会把接了 goal_pos 的 75% 和
    # 没接的 6% 平均成 29±40%，一个不对应任何实验的数字。
    # goal_slot 也算"给了目标"：有 goal_pos 的任务照常拿到它，没有的任务拿到
    # 一个显式置零并标了 valid=0 的槽，与"根本没这一路输入"不是一回事。
    has_goal_input = df.use_goal | df.goal_slot
    df["method"] = df.method.where(has_goal_input, df.method + "/nogoal")
    # 同一 checkpoint+配置可能被评测多次（主评测、步数消融的 10 步、CFG 消融的
    # w=1.0 会落在同一个键上）。这些是对同一个量的重复测量，**取平均**而不是
    # 任选一次 —— 实测同配置三次得到 4% / 5% / 7%，选最后一次等于让结果表取决于
    # 脚本的执行顺序。n_episodes 也进分组键，样本量不同的测量不会被混在一起。
    key = ["ckpt", "task", "use_goal", "goal_slot", "n_steps", "guidance",
           "execute_horizon", "ema", "method", "model", "seed", "run", "multitask",
           "n_episodes"]
    g = df.groupby(key, as_index=False).agg(
        success_rate=("success_rate", "mean"),
        spread=("success_rate", lambda x: x.max() - x.min()),
        n_evals=("success_rate", "size"))
    rep = g[g.n_evals > 1]
    if not rep.empty:
        worst = rep.spread.max()
        print(f"注意：{int(rep.n_evals.sum())} 次评测被合并为 {len(rep)} 个配置的均值；"
              f"同配置重复测量的最大跨度 {worst:.0%}（这是成功率的测量噪声下限）\n")
    return g


def agg(df):
    """按 (方法, 任务) 聚合多个 seed，给出 mean±std。"""
    g = df.groupby(["method", "task"])["success_rate"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def main_table(df: pd.DataFrame) -> str:
    """主结果表：各方法 × 各任务的成功率。"""
    a = agg(main_rows(df))
    tasks = [t for t in SHORT if t in set(a.task)]
    lines = ["| Method | " + " | ".join(SHORT[t] for t in tasks) + " |",
             "|---|" + "---|" * len(tasks)]
    for m in order_of(set(a.method)):
        sub = a[a.method == m]
        if sub.empty:
            continue
        cells = []
        for t in tasks:
            r = sub[sub.task == t]
            cells.append(f"{100*r['mean'].iloc[0]:.0f}±{100*r['std'].iloc[0]:.0f}%"
                         if not r.empty else "—")
        label = LABEL.get(m, LABEL.get(m.split("/")[0], m) + " · online")
        name = f"**{label}**" if m == "fm" else label
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_all(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    # --- 图1：主结果分组柱状图 ---
    a = agg(main_rows(df))
    tasks = [t for t in SHORT if t in set(a.task)]
    models = order_of(set(a.method))
    if tasks and models:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            sub = a[a.method == m].set_index("task")
            xs = [j + i * w - 0.4 + w / 2 for j in range(len(tasks))]
            ys = [100 * sub.loc[t, "mean"] if t in sub.index else 0 for t in tasks]
            es = [100 * sub.loc[t, "std"] if t in sub.index else 0 for t in tasks]
            ax.bar(xs, ys, w * 0.9, yerr=es, capsize=3,
                   label=LABEL.get(m, m), color=COLOR.get(m, "#666"))
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([SHORT[t] for t in tasks])
        ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
        ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.set_title("ManiSkill3 success rate (100 episodes)")
        plt.tight_layout(); p = out_dir / "results_main.png"
        plt.savefig(p, dpi=140); figs.append(p); plt.close()

    # --- 图2：采样步数 vs 成功率 ---
    sw = (df[df.guidance.eq(1.0) & df.execute_horizon.eq(DEFAULT_HORIZON)]
          .groupby(["method", "task", "n_steps"])["success_rate"].mean().reset_index())
    if sw.n_steps.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, t), grp in sw.groupby(["method", "task"]):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("n_steps")
            ax.plot(grp.n_steps, 100 * grp.success_rate, "o",
                    color=COLOR.get(m, "#666"),
                    ls="--" if m == "ddpm" else "-",
                    label=f"{LABEL.get(m,m)} · {SHORT.get(t,t)}")
        if ax.get_lines():
            ax.set_xscale("log"); ax.set_xlabel("Sampling steps")
            ax.set_ylabel("Success rate (%)"); ax.grid(alpha=0.3)
            ax.legend(fontsize=8); ax.set_title("Quality vs. sampling steps")
            plt.tight_layout(); p = out_dir / "results_steps.png"
            plt.savefig(p, dpi=140); figs.append(p)
        plt.close()

    # --- 图3：CFG 权重消融 ---
    gw = (df[df.guidance.notna() & df.execute_horizon.eq(DEFAULT_HORIZON)]
          .groupby(["task", "guidance"])["success_rate"].mean().reset_index())
    if gw.guidance.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for t, grp in gw.groupby("task"):
            grp = grp.sort_values("guidance")
            ax.plot(grp.guidance, 100 * grp.success_rate, "o-", label=SHORT.get(t, t))
        if ax.get_lines():
            ax.set_xlabel("CFG guidance weight w"); ax.set_ylabel("Success rate (%)")
            ax.grid(alpha=0.3); ax.legend(); ax.set_title("Classifier-free guidance ablation")
            plt.tight_layout(); p = out_dir / "results_cfg.png"
            plt.savefig(p, dpi=140); figs.append(p)
        plt.close()

    return figs


README_BEGIN = "<!-- RESULTS_TABLE -->"
README_END = "<!-- /RESULTS_TABLE -->"


def write_readme(table: str, figs: list, readme: Path, weights: str) -> None:
    """把结果表写进 README 的标记区间，幂等：重复运行只替换区间内容。"""
    text = readme.read_text()
    assert README_BEGIN in text, f"README 里找不到 {README_BEGIN} 标记"

    body = [table, ""]
    for f in figs:
        rel = f.as_posix()
        body.append(f"![{f.stem}]({rel})")
    body.append("")
    body.append(f"*成功率为 100 episode 的评测结果，多 seed 取 mean±std；"
                f"推理权重：{'EMA' if weights == 'ema' else '在线（非 EMA）'}。"
                f"复现：`python scripts/make_report.py --weights {weights}`*")
    block = f"{README_BEGIN}\n\n" + "\n".join(body) + f"\n\n{README_END}"

    if README_END in text:
        head, _, rest = text.partition(README_BEGIN)
        _, _, tail = rest.partition(README_END)
        text = head + block + tail
    else:
        text = text.replace(README_BEGIN, block)
    readme.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    # 最终评测对每个 checkpoint 同时跑了 EMA 与在线权重。两者绝不能混在一起
    # 聚合——那等于把两个不同的推理配置平均掉。默认只报 EMA。
    # 默认是 online 而不是通行的 EMA。本项目实测 ema_decay=0.9999 的时间常数
    # (10000 步) 对 20000 步的训练太慢：同一批 checkpoint 在线权重是
    # 6/37/59/70/75%，EMA 读数却是平的 8/4/4/4/4%。见 docs/next-steps.md。
    ap.add_argument("--weights", choices=("ema", "online", "both"), default="online",
                    help="用哪套权重的评测结果出表（默认 online，理由见代码注释）")
    ap.add_argument("--write-readme", action="store_true",
                    help=f"把结果表和图写进 README 的 {README_BEGIN} 标记区间")
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["success_rate"] = df.success_rate.astype(float)
    n_raw = len(df)
    df = prepare(df)
    if args.weights == "ema":
        df = df[df.ema]
    elif args.weights == "online":
        df = df[~df.ema]
    else:
        df = df.copy()
        df["method"] = df.method.where(df.ema, df.method + "/online")
    assert not df.empty, f"CSV 里没有 weights={args.weights} 的评测记录"
    print(f"{n_raw} 条评测记录，去重并按 weights={args.weights} 筛选后 {len(df)} 条\n")

    # 主结果表的每一格都标称"100 episode"。混入不同 episode 数意味着表里并排的
    # 数字来自不同可信度的测量 —— 通常是上一轮遗留在 results.csv 里的行。
    base = main_rows(df)
    if base.n_episodes.nunique() > 1:
        counts = base.n_episodes.value_counts().to_dict()
        print(f"警告：主结果表混入了不同的 episode 数 {counts}，"
              f"并排的数字可信度不一致。多半是上一轮的遗留行，请检查 {args.csv}\n")
    print("主结果表：\n")
    print(main_table(df))
    figs = plot_all(df, args.out_dir)
    print("\n生成的图：")
    for f in figs:
        print(f"  {f}")
    if args.write_readme:
        write_readme(main_table(df), figs, args.readme, args.weights)
        print(f"\n已写入 {args.readme}")


if __name__ == "__main__":
    main()
