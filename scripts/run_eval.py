"""从 checkpoint 做完整评测，输出 CSV 供汇总成结果表。

    python scripts/run_eval.py --ckpt outputs/fm_PickCube_s42/ckpt_20000.pt \
        --tasks PickCube-v1 --n-episodes 100

    # 采样步数消融（同一个 checkpoint，不用重训）
    python scripts/run_eval.py --ckpt ... --sweep-steps 1 2 5 10 20

    # CFG 权重消融
    python scripts/run_eval.py --ckpt ... --sweep-guidance 1.0 1.5 2.0

    # 执行长度消融（一次预测执行几步再重规划）——与采样步数一起决定可达控制频率
    python scripts/run_eval.py --ckpt ... --sweep-horizon 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_env, obs_kwargs, rollout  # noqa: E402
from src.models.policy import (BCPolicy, BCXAttnPolicy, DDPMPolicy,  # noqa: E402
                               FMPolicy, load_trainable_state_dict)

BUILDERS = {"fm": FMPolicy, "bc": BCPolicy, "bc_xattn": BCXAttnPolicy,
            "ddpm": DDPMPolicy}


def load_policy(ckpt_path: Path, use_ema: bool = True, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    name = cfg["model"]["name"]
    common = dict(act_dim=cfg.get("act_dim", 8), act_horizon=cfg["act_horizon"],
                  img_size=cfg["img_size"],
                  proprio_dim=cfg.get("proprio_dim", 9), use_proprio=cfg["use_proprio"],
                  d_model=cfg["model"]["d_model"])
    if name == "bc":
        # hidden 必须从 checkpoint 的 cfg 读。等容量对照用的是 hidden=2364，
        # 用默认值 1024 建模型会在 load_state_dict 时炸出 shape mismatch。
        # 凡是进了模型构造函数的超参，这里都得跟着读，否则加一个配置项就断一次。
        policy = BCPolicy(hidden=cfg["model"].get("hidden", 1024), **common)
    elif name == "bc_xattn":
        policy = BCXAttnPolicy(
            n_layers=cfg["model"]["n_layers"], n_heads=cfg["model"]["n_heads"],
            global_cond=cfg.get("global_cond", False), **common)
    else:
        gen = dict(n_layers=cfg["model"]["n_layers"], n_heads=cfg["model"]["n_heads"],
                   cfg_dropout=cfg["model"].get("cfg_dropout", 0.2),
                   global_cond=cfg.get("global_cond", False), **common)
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


# 每次给结果表加一列，旧文件就少一列。直接追加会让两种 schema 混在一个文件里，
# 而 pandas 读出来的错位**不会报错**，只会安静地把数字对到错误的列上。
# 表里的每一列都是"某次把不同实验平均成一个假数字"之后加上去的，见 docs/debugging.md，
# 所以这里宁可原地迁移，也不允许两种表头共存。
MIGRATIONS = (
    ("goal_slot", "False"),   # 加列前的评测都不是 goal_slot 观测
    ("n_average", "1"),       # 加列前都是抽单个样本
    ("val_frac", "0.0"),      # 加列前没有留出集，全部 episode 进训练
    ("max_steps", "300"),     # 加列前评测一律 300 步上限
)


def migrate_csv_header(path, header, migrations=MIGRATIONS):
    """把旧表头就地补齐到 `header`，补不齐就直接报错。

    只处理"恰好少一列"的情况：少两列以上说明文件来自更早的年代，
    与其猜测每一列该填什么，不如让调用方换一个新文件。
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    existing = rows[0] if rows else []
    for missing, default in migrations:
        if existing != [c for c in header if c != missing]:
            continue
        print(f"  {path} 缺 {missing} 列，就地补上（旧行一律 {default}）", flush=True)
        Path(path).with_suffix(".csv.bak").write_bytes(Path(path).read_bytes())
        j = header.index(missing)
        with open(path, "w", newline="") as fh2:
            w = csv.writer(fh2)
            w.writerow(header)
            for r in rows[1:]:
                w.writerow(r[:j] + [default] + r[j:])
        existing = header
        break
    assert existing == header, (
        f"{path} 的表头是旧格式 {existing}，与当前不符。\n"
        f"用 --out 指定一个新文件，不要混写。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--sweep-steps", type=int, nargs="+", default=None)
    ap.add_argument("--sweep-guidance", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-horizon", type=int, nargs="+", default=None,
                    help="execute_horizon：一次预测执行几步再重规划")
    ap.add_argument("--n-average", type=int, default=1,
                    help="独立采样 n 次取平均（用蒙特卡洛估计条件均值而不是抽一个样本）")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="每个 episode 的环境步数上限。ManiSkill 给 PickCube-v1 "
                         "注册的是 50，官方基线用的是 100，本项目一直用 300。"
                         "这个值改成绩，所以它进结果表的 schema。")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/results.csv"))
    args = ap.parse_args()

    policy, cfg, name = load_policy(args.ckpt, use_ema=not args.no_ema)
    tasks = args.tasks or cfg["tasks"]
    # BC 一次前向就出动作块，没有采样步数可言。记成 1 而不是继承一个 10，
    # 否则 CSV 里 BC 会看起来像跑了 10 步去噪。
    default_steps = cfg["model"].get("n_sample_steps",
                                     1 if name.startswith("bc") else 10)
    steps_list = args.sweep_steps or [default_steps]
    guid_list = args.sweep_guidance or [1.0]
    horizon_list = args.sweep_horizon or [cfg["eval"]["execute_horizon"]]

    print(f"checkpoint: {args.ckpt}  模型={name}  EMA={not args.no_ema}")
    print(f"任务={tasks}  episodes={args.n_episodes}\n")

    # use_goal 必须进 CSV。它是观测配置，不是超参：同一个 PickCube，接不接
    # goal_pos 的成功率是 6% 和 75%。不记下来，汇总脚本会把两者按 (方法,任务)
    # 平均成 "29±40%" —— 一个不对应任何实验的数字。整个项目最贵的一课就是
    # 这个，所以它固化在结果表的 schema 里，而不是留在人的记忆里。
    # goal_slot 与 use_goal 是两种不同的观测配置，必须分开记：
    #   use_goal   proprio = qpos(9) + goal(3)，只有带显式目标的任务能开
    #   goal_slot  proprio = qpos(9) + goal(3) + valid(1)，所有任务统一，
    #              没有 goal_pos 的任务填零并置 valid=0
    # 两者混进一列，汇总时就会把"给了目标"和"给了空目标槽"平均成一个数。
    # n_average 也必须进 schema。它不是"同一个配置的重复测量"，是**另一个推理
    # 配置**：K 个样本取平均估计条件均值，与抽一个样本是两回事（实测 PickCube
    # K=1/4/16 → 60/56/53%）。不记这一列，汇总脚本会把三者按 (ckpt,任务,步数)
    # 平均成 56%，一个不对应任何实验的数字。这与 use_goal 那一课是同一个形状。
    # 训练配置也必须进 schema，理由与 use_goal 完全相同，而且这一条已经在出错：
    # outputs/2026-09-11/multitask.csv 里有 11 个 run，它们在 task_goal、训练步数、
    # 模型宽度、cfg_dropout、OT 耦合上各不相同，但这些全都不在表里。于是 model="fm"
    # 的八个 run 被 make_report 当成同一个方法的八个 seed 平均成 "4±2%"，
    # 而 README 里那一行是 60/96/36。多 seed 实验会让这件事变得更糟：
    # 真正的 seed 重复与不同配置的 run 在表里长得一模一样。
    #   task_goal    每任务取目标 actor，与 use_goal/goal_slot 是三种不同的观测配置
    #   steps        训练预算。20000 与 60000 步的多任务不是同一个实验
    #   d_model      容量。6.80M 与 15.01M 的结论相反（60/96/36 对 4/95/8）
    #   cfg_dropout  训练时指令置空的比例。0.2 与 0 实测 63/90/7 对 60/96/36
    #   ot_coupling  batch 内最优指派。多任务上是负结果（2/96/0）
    #   global_cond  非视觉 token 经 AdaLN 再注入一次
    #   sources      演示来源。混合来源是另一份数据，不是另一个 seed
    #   val_frac     留出比例。少 10% 训练数据是不同的实验
    header = ["ckpt", "model", "seed", "task", "use_goal", "goal_slot", "task_goal",
              "val_frac", "steps", "d_model", "cfg_dropout", "ot_coupling",
              "global_cond", "sources", "n_steps", "guidance", "execute_horizon",
              "max_steps", "n_average", "ema", "n_episodes", "success_rate",
              "mean_length"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    if not new_file:
        migrate_csv_header(args.out, header)
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(header)

        for task in tasks:
            env = make_env(task)
            try:
                for s in steps_list:
                    for g in guid_list:
                        for h in horizon_list:
                            t0 = time.perf_counter()
                            r = rollout(policy, env, task, n_episodes=args.n_episodes,
                                        obs_horizon=cfg["obs_horizon"], execute_horizon=h,
                                        img_size=cfg["img_size"], n_steps=s, guidance=g,
                                        max_steps=args.max_steps,
                                        n_average=args.n_average, **obs_kwargs(cfg))
                            print(f"  {task:22s} steps={s:<4} w={g:<4} H={h:<3} "
                                  f"T={args.max_steps:<4} "
                                  f"SR {100*r['success_rate']:5.1f}%  "
                                  f"步长 {r['mean_length']:5.1f}  "
                                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
                            mcfg = cfg.get("model", {})
                            w.writerow([args.ckpt.as_posix(), name, cfg["seed"], task,
                                        cfg.get("use_goal", False),
                                        cfg.get("goal_slot", False),
                                        cfg.get("task_goal", False),
                                        cfg.get("val_frac", 0.0),
                                        cfg.get("steps", ""),
                                        mcfg.get("d_model", ""),
                                        mcfg.get("cfg_dropout", ""),
                                        mcfg.get("ot_coupling", False),
                                        cfg.get("global_cond", False),
                                        "+".join(cfg.get("sources",
                                                         ["motionplanning"])),
                                        s, g, h, args.max_steps,
                                        args.n_average, not args.no_ema, args.n_episodes,
                                        f"{r['success_rate']:.4f}", f"{r['mean_length']:.2f}"])
                            fh.flush()
            finally:
                env.close()
    print(f"\n结果已追加到 {args.out}")


if __name__ == "__main__":
    main()
