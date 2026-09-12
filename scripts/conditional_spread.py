"""数据本身的条件动作分布有多宽？

选生成式动作头（Flow Matching / Diffusion）而不是回归，理由是"同一个观测可能
对应多种合法动作，MSE 回归会塌到条件均值，而均值可能本身就非法"。这个理由
**在一份具体的数据上成不成立，是可以量的**，而且必须先量再选模型。

`action_diversity.py` 量的是**模型**的采样散度；这里量的是**数据**的条件散度：
在真值状态空间里找近邻，看**不同演示**在同一个情境下的动作分歧有多大。

为什么用真值状态而不是视觉特征：视觉编码器本身就是个变量，用它当距离会把
"编码器分不开这两个情境"混进"这两个情境的动作确实不同"里。真值状态
（末端位置、被操作物位置、目标位置、夹爪开合）没有这个问题。

核心的方法学问题：散度随近邻半径单调上升 —— 半径放大，"同一个情境"就变松，
动作当然更不一样。所以逐半径分箱报数，看半径 → 0 时的截距，
那才是条件散度的估计。只报一个数字是没有意义的。

近邻一律取自**不同的 episode**：同一条轨迹相邻时刻的动作天然相似，
把它们算进来会把条件散度压到接近 0，得出"数据是确定性的"这个假结论。

    python scripts/conditional_spread.py --task PickCube-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import TASK_GOAL_ACTOR, default_h5_path  # noqa: E402

# 被操作物在不同任务里的 actor 名
MANIP_ACTOR = {"PickCube-v1": "cube", "PushCube-v1": "cube",
               "PullCube-v1": "cube", "StackCube-v1": "cubeA"}


def load_task(task: str, max_episodes: int | None, act_horizon: int,
              source: str = "motionplanning", ep_offset: int = 0):
    """把状态、动作块、episode 归属一次读进内存（都很小，图像不读）。"""
    path = default_h5_path(task, source=source)
    goal_actor, manip_actor = TASK_GOAL_ACTOR[task], MANIP_ACTOR[task]
    states, chunks, ep_ids = [], [], []
    with h5py.File(path, "r") as f:
        names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
        if max_episodes:
            names = names[:max_episodes]
        for ei, name in enumerate(names):
            ep = f[name]
            if not bool(np.asarray(ep["success"])[-1]):
                continue
            # 与 ManiskillDataset 一致：环境执行前会把动作裁到 [-1,1]，
            # 演示里记录的却是裁剪前的原始输出（PPO 的演示 17.9% 越界）。
            acts = np.clip(np.asarray(ep["actions"], dtype=np.float32), -1.0, 1.0)
            T = acts.shape[0]
            tcp = np.asarray(ep["obs"]["extra"]["tcp_pose"])[:T, :3]
            obj = np.asarray(ep["env_states"]["actors"][manip_actor])[:T, :3]
            goal = np.asarray(ep["env_states"]["actors"][goal_actor])[:T, :3]
            grip = np.asarray(ep["obs"]["agent"]["qpos"])[:T, -1:]
            # 状态 = 末端在哪 + 物体在哪 + 目标在哪 + 夹爪开合。
            # 夹爪不可省：抓取前后末端与物体可以在同一位置，但该做的事完全不同。
            states.append(np.concatenate([tcp, obj, goal, grip], axis=1))
            # 动作块，末尾不足时重复最后一个动作（与 ManiskillDataset 一致）
            idx = np.minimum(np.arange(T)[:, None] + np.arange(act_horizon)[None, :], T - 1)
            chunks.append(acts[idx])                                   # (T, H, A)
            ep_ids.append(np.full(T, ei + ep_offset, dtype=np.int32))
    return (np.concatenate(states), np.concatenate(chunks), np.concatenate(ep_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--sources", nargs="+", default=["motionplanning"],
                    help="演示来源目录；给多个就是混合数据（用来制造真正的多模态）")
    ap.add_argument("--act-horizon", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--k", type=int, default=8, help="每个 query 取多少个近邻")
    ap.add_argument("--pool", type=int, default=30000, help="近邻候选池大小")
    ap.add_argument("--balanced-sources", action="store_true",
                    help="强制从每个来源各取 k/2 个近邻。检验'同一个情境下两种解法"
                         "是否不同'必须这样做 —— 否则最近邻几乎都来自同一个来源"
                         "（两种解法在状态空间里各自成团），每组仍是单峰的，"
                         "混合数据看起来反而比单一来源更单峰。")
    args = ap.parse_args()

    parts, src_ids = [], []
    off = 0
    for si, src in enumerate(args.sources):
        st, ch, ei = load_task(args.task, args.max_episodes, args.act_horizon,
                               source=src, ep_offset=off)
        off = int(ei.max()) + 1
        parts.append((st, ch, ei))
        src_ids.append(np.full(len(st), si, dtype=np.int8))
        print(f"  来源 {src:16s} {len(st):>8,} 帧 / {len(np.unique(ei)):>4} 条轨迹")
    states = np.concatenate([p[0] for p in parts])
    chunks = np.concatenate([p[1] for p in parts])
    ep_ids = np.concatenate([p[2] for p in parts])
    src_of = np.concatenate(src_ids)
    rng = np.random.default_rng(0)
    if len(states) > args.pool:
        sel = rng.choice(len(states), args.pool, replace=False)
        states, chunks, ep_ids, src_of = states[sel], chunks[sel], ep_ids[sel], src_of[sel]

    # 动作按边缘统计量归一化，于是"边缘散度 = 1"，与 action_diversity.py 同一口径
    flat = chunks.reshape(-1, chunks.shape[-1])
    a_mean, a_std = flat.mean(0), flat.std(0).clip(1e-6)
    norm = (chunks - a_mean) / a_std

    # 状态各维按自身标准差标准化，避免"米"和"弧度"混在一个欧氏距离里
    s_std = states.std(0).clip(1e-6)
    S = states / s_std

    qi = rng.choice(len(S), min(args.n_queries, len(S)), replace=False)
    rows, projections, top_pc_frac = [], [], []
    for i in qi:
        d = np.linalg.norm(S - S[i], axis=1)
        d[ep_ids == ep_ids[i]] = np.inf        # 只看别的演示怎么做
        if args.balanced_sources and len(args.sources) > 1:
            # 每个来源各取 k/2 个，于是这一组里两种解法各占一半 ——
            # 这才是"同一个情境下两种解法是否不同"该有的样本
            per = max(1, args.k // len(args.sources))
            picks = []
            for si in range(len(args.sources)):
                dd = np.where(src_of == si, d, np.inf)
                cand = np.argpartition(dd, per)[:per]
                picks.append(cand[np.isfinite(dd[cand])])
            nn = np.concatenate(picks)
        else:
            nn = np.argpartition(d, args.k)[:args.k]
            nn = nn[np.isfinite(d[nn])]
        if len(nn) < 2:
            continue
        group = norm[np.concatenate([[i], nn])]            # (k+1, H, A)
        spread = np.sqrt((group.std(axis=0) ** 2).mean())  # 与 sample_spread 同定义
        # 散度大 ≠ 多模态。把这一组动作块投到它们自己的主方向上，
        # 攒起来看这个一维分布是单峰还是双峰 —— 这是"回归会塌到条件均值"
        # 那个论证真正依赖的性质，而散度本身区分不了"几个模态"和"一个宽的单峰"。
        g = group.reshape(len(group), -1)
        g = g - g.mean(0)
        if g.shape[0] > 2:
            u, sv, _ = np.linalg.svd(g, full_matrices=False)
            proj = u[:, 0] * sv[0]
            sd = proj.std()
            if sd > 1e-8:
                projections.append(proj / sd)
                top_pc_frac.append(sv[0] ** 2 / max((sv ** 2).sum(), 1e-12))
        # 半径用物理量报：末端与物体位置的距离，单位毫米，读者能判断"多同算同"
        phys = np.linalg.norm(states[nn][:, :6] - states[i, :6], axis=1).mean() * 1000
        rows.append((phys, spread))

    rows = np.array(rows)
    print(f"\n===== 数据的条件动作散度：{args.task} / {'+'.join(args.sources)} =====")
    print(f"候选池 {len(S):,} 帧 / {len(np.unique(ep_ids))} 条轨迹，"
          f"{len(rows)} 个 query × {args.k} 个近邻（均取自其它演示）")
    print(f"动作按边缘统计量归一化，所以**边缘散度按定义为 1.000**\n")
    print(f"  {'近邻半径 (mm)':<18}{'query 数':>9}{'条件散度':>12}{'占边缘':>10}")
    edges = [0, 10, 20, 40, 80, 160, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum() < 10:
            continue
        lab = f"{lo:.0f}–{hi:.0f}" if np.isfinite(hi) else f">{lo:.0f}"
        print(f"  {lab:<18}{m.sum():>9}{np.median(rows[m, 1]):>12.3f}"
              f"{100*np.median(rows[m,1]):>9.0f}%")
    print(f"\n  全部 query 的中位数 {np.median(rows[:,1]):.3f}")

    # ---- 这个散度是多模态还是单模态的宽分布 ----
    pj = np.concatenate(projections)
    m2 = ((pj - pj.mean()) ** 2).mean()
    skew = ((pj - pj.mean()) ** 3).mean() / m2 ** 1.5
    kurt = ((pj - pj.mean()) ** 4).mean() / m2 ** 2
    bc = (skew ** 2 + 1) / kurt
    print(f"\n  近邻动作投到各自主方向后（{len(pj):,} 个点，逐组标准化）：")
    print(f"    主方向占总方差 {100*np.median(top_pc_frac):.0f}%")
    print(f"    偏度 {skew:+.2f}  峰度 {kurt:.2f}  双峰系数 {bc:.3f}"
          f"（正态分布是 0.555，>0.555 提示双峰）")
    print("    " + ("→ 提示多模态：条件均值可能落在两个模态之间，回归会给出非法动作。"
                    if bc > 0.555 else
                    "→ **单峰**。散度是一个宽的单峰，不是多个模态 —— "
                    "条件均值本身就是一个合法动作，"))
    if bc <= 0.555:
        print("       '回归会塌到条件均值因而失败'这个选生成模型的理由，"
              "在这份数据上不成立。")
    print("\n判读：散度随半径单调上升是必然的（半径放大，'同一个情境'就变松），"
          "\n      所以要看最小半径那一档，它是条件散度的上界估计。"
          "\n      但**散度本身不回答选不选生成模型** —— 决定性的是上面那个双峰检验："
          "\n      散度区分不了'几个模态'和'一个宽的单峰'，而'回归会塌到条件均值'"
          "\n      这个论证依赖的恰恰是前者。")
    if len(args.sources) > 1 and not args.balanced_sources:
        print("\n提醒：混合了多个来源但没开 --balanced-sources。最近邻会几乎全部落在"
              "\n      同一个来源里（不同解法在状态空间里各自成团），于是每一组仍然是"
              "\n      单峰的，混合数据看起来反而比单一来源更单峰。要检验'同一情境下"
              "\n      两种解法是否不同'，必须加 --balanced-sources。")


if __name__ == "__main__":
    main()
