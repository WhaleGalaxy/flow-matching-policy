"""留出集划分的回归测试。

这里守的全是**不会报错**的失败：留出集混进了训练过的 episode，量出来的
"泛化误差"只会偏小，不会有任何异常表现，然后一路写进 README。所以每一条
不变量都要钉死在测试里，而不是靠注释和记忆。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import (ManiskillDataset, default_h5_path,  # noqa: E402
                             split_episodes)

NAMES = [f"traj_{i}" for i in range(200)]


def test_train_and_val_are_disjoint_and_cover_everything():
    tr = split_episodes(NAMES, val_frac=0.1, split_seed=0, split="train")
    va = split_episodes(NAMES, val_frac=0.1, split_seed=0, split="val")
    assert set(tr) & set(va) == set(), "训练集与留出集有交集"
    assert set(tr) | set(va) == set(NAMES), "有 episode 两边都没进"
    assert len(va) == 20


def test_split_depends_only_on_split_seed():
    """划分必须与训练 seed 无关。

    多 seed 实验要量的是"换一个初始化结论还成不成立"。划分要是跟着训练 seed 变，
    量出来的方差里就混进了"换一批数据"，多 seed 本来要回答的问题反而答不了；
    方法之间的对照也不再是同一份训练集。
    """
    a = split_episodes(NAMES, 0.1, split_seed=0, split="val")
    b = split_episodes(NAMES, 0.1, split_seed=0, split="val")
    c = split_episodes(NAMES, 0.1, split_seed=1, split="val")
    assert a == b, "同一个 split_seed 给出了不同的划分"
    assert a != c, "换了 split_seed 划分却没变，划分没有真正用上这个参数"


def test_val_frac_zero_keeps_the_old_behaviour():
    """val_frac=0 是 2026-09-12 之前所有结果的配置，必须原样可复现。"""
    assert split_episodes(NAMES, 0.0, split="train") == NAMES
    assert split_episodes(NAMES, 0.0, split="val") == []


def test_split_preserves_episode_order():
    """返回顺序与输入一致，日志里的 episode 编号才对得上。"""
    tr = split_episodes(NAMES, 0.1, split_seed=3, split="train")
    assert tr == [n for n in NAMES if n in set(tr)]


@pytest.mark.parametrize("val_frac", [1.0, 2.0, 0.0001])
def test_unusable_split_fails_loudly(val_frac):
    """切不出可用划分时要直接报错，而不是安静地给一个空集或空训练集。

    0.0001 这一档是真会踩的：200 条轨迹上四舍五入到 0 条留出，
    于是 val_frac 看起来设了、留出集其实是空的。"""
    with pytest.raises(AssertionError):
        split_episodes(NAMES, val_frac, split="val")


def test_rejects_unknown_split():
    with pytest.raises(AssertionError):
        split_episodes(NAMES, 0.1, split="test")


# ---------------------------------------------------------- 真实数据上的接线

def _has_demos(task="PickCube-v1"):
    return default_h5_path(task).exists()


@pytest.mark.skipif(not _has_demos(), reason="本机没有重放好的演示数据")
def test_dataset_train_and_val_share_no_episode():
    """接线测试：纯函数对不代表 Dataset 用对了。

    真正会出事的是 Dataset 里"先筛后切"的顺序 —— 先切再按 success_only 筛，
    两个划分的比例就会随失败轨迹的分布漂移，而且不会报错。
    """
    task = "PickCube-v1"
    kw = dict(task=task, val_frac=0.1, split_seed=0, max_episodes=60)
    tr = ManiskillDataset(default_h5_path(task), split="train", **kw)
    va = ManiskillDataset(default_h5_path(task), split="val", **kw)
    assert set(tr.episode_lengths) & set(va.episode_lengths) == set()
    assert len(va.episode_lengths) > 0 and len(tr.episode_lengths) > 0
    # 索引也不能漏：两个划分的样本数加起来等于不切时的总数
    allds = ManiskillDataset(default_h5_path(task), split="all", **kw)
    assert len(tr) + len(va) == len(allds)


@pytest.mark.skipif(not _has_demos(), reason="本机没有重放好的演示数据")
def test_normalizer_statistics_come_from_train_split_only():
    """归一化统计量是从 dataset 上采的，所以它跟着划分走。

    留出集的动作参与了归一化统计，等于把留出集的信息漏进了训练。
    量级很小，但这正是"不会报错"的那一类问题。
    """
    task = "PickCube-v1"
    kw = dict(task=task, val_frac=0.1, split_seed=0, max_episodes=60)
    tr = ManiskillDataset(default_h5_path(task), split="train", **kw)
    picks = {tr.index[i][0] for i in range(len(tr.index))}
    va = ManiskillDataset(default_h5_path(task), split="val", **kw)
    assert picks.isdisjoint(set(va.episode_lengths))


# ---------------------------------------------------------------- 发散看门狗

def test_divergence_watchdog_catches_the_real_run_that_was_missed():
    """用 2026-09-12 那次真实发散的验证 loss 序列回放。

    那一轮跑满了 60000 步、存了 checkpoint、还被评测了三个任务，全程没有任何
    报错，白花一小时。看门狗必须在验证 loss 第一次明显反弹之后就抓到它。
    """
    from src.train import diverging
    # 真实读数（train_fm_mt_val10_s7.log），每 5000 步一个
    curve = [(2500, 0.3255), (7500, 0.2373), (12500, 0.2100), (17500, 0.1983),
             (22500, 0.1797), (27500, 0.1599), (32500, 0.1577), (37500, 0.2771),
             (42500, 0.5845), (47500, 0.7705), (52500, 0.8085), (57500, 0.7204)]
    best, bad, caught_at = float("inf"), 0, None
    for step, vl in curve:
        bad = diverging(vl, best, bad, step, 60000)
        best = min(best, vl)
        if bad >= 2:
            caught_at = step
            break
    assert caught_at == 42500, f"应在第 42500 步抓到，实际 {caught_at}"


def test_divergence_watchdog_does_not_fire_on_a_healthy_run():
    """seed 42 那一轮是正常的，看门狗不能误伤 —— 误报一次就没人再信它。"""
    from src.train import diverging
    curve = [(2500, 0.3179), (10000, 0.2090), (20000, 0.1666), (30000, 0.1374),
             (40000, 0.1616), (50000, 0.1540), (60000, 0.1510)]
    best, bad = float("inf"), 0
    for step, vl in curve:
        bad = diverging(vl, best, bad, step, 60000)
        best = min(best, vl)
        assert bad < 2, f"在第 {step} 步误判为发散（loss {vl}）"


def test_divergence_watchdog_is_silent_during_early_training():
    """前 20% loss 本来就在剧烈下降，最低点还没稳定，这段不该生效。"""
    from src.train import diverging
    assert diverging(vl=99.0, best=0.1, bad=1, step=1000, total=60000) == 0
