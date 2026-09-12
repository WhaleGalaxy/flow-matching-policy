"""分析与汇总代码的单元测试。

这些测试守的都是**安静出错**的地方：探针给出一个看似合理但没有意义的 R²、
汇总脚本把两个不同实验的数字平均成一个不对应任何实验的数字。
它们不会抛异常，只会让 README 里出现一个错的表。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.diagnostics import linear_probe  # noqa: E402


def _load(name: str):
    """脚本目录不是包，按路径加载。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 线性探针

def _low_rank_features(n=400, d=3000, rank=30, seed=0):
    """真实 ViT 特征是强相关的低有效秩，用低秩合成才有代表性。"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, rank))
    return z, z @ rng.normal(size=(rank, d)) + 0.1 * rng.normal(size=(n, d))


def test_probe_finds_information_that_is_present():
    z, x = _low_rank_features()
    y = z[:, :3] @ np.random.default_rng(1).normal(size=(3, 2))
    assert linear_probe(x, y)["r2"].min() > 0.9


def test_probe_rejects_information_that_is_absent():
    """留出 R² 必须能变成 ~0。这是整个观测充分性结论的支点：
    如果探针对无关目标也给出高 R²，"目标不可观测"就无从谈起。"""
    _, x = _low_rank_features()
    y = np.random.default_rng(2).normal(size=(len(x), 2))
    assert linear_probe(x, y)["r2"].max() < 0.1


def test_probe_reports_holdout_not_train_fit():
    """特征维度远多于样本数时，训练集上什么都能拟合。若实现退化成报训练集
    R²，上面那个"无关目标"的测试会得到 ≈1 —— 这里直接钉住维度关系。"""
    _, x = _low_rank_features(n=120, d=5000)
    y = np.random.default_rng(3).normal(size=(120, 1))
    r = linear_probe(x, y)
    assert r["n_train"] + r["n_test"] == 120 and r["n_test"] > 0
    assert r["r2"][0] < 0.3


# ---------------------------------------------------------------- 部署包络

def test_envelope_is_monotone_and_drops_dominated_points():
    pareto = _load("pareto")
    pts = [
        (100.0, 0.75, ("a",)),   # 慢而准
        (400.0, 0.60, ("b",)),
        (800.0, 0.30, ("c",)),
        (200.0, 0.40, ("d",)),   # 被 b 支配：更慢且更差
    ]
    env = pareto.envelope(pts)
    tags = [m[0] for _, _, m in env]
    assert "d" not in tags, "被支配的配置不应留在包络上"
    freqs = [f for f, _, _ in env]
    srs = [s for _, s, _ in env]
    assert freqs == sorted(freqs)
    assert srs == sorted(srs, reverse=True), "频率越高，可达成功率必须不增"


# ---------------------------------------------------------------- 失败归因

def test_failure_stages_take_the_first_unmet_link():
    fm = _load("failure_modes")
    assert fm.classify(dict(success=True, placed=True, grasped=True, dmin=0.01)) == "成功"
    assert fm.classify(dict(success=False, placed=True, grasped=True, dmin=0.01)) == "送达未静止"
    assert fm.classify(dict(success=False, placed=False, grasped=True, dmin=0.01)) == "抓取未送达"
    assert fm.classify(dict(success=False, placed=False, grasped=False, dmin=0.01)) == "接近未抓取"
    assert fm.classify(dict(success=False, placed=False, grasped=False, dmin=0.30)) == "未接近"


# ---------------------------------------------------------------- 结果汇总

def _row(**kw):
    base = dict(ckpt="outputs/run_s42/ckpt_20000.pt", model="fm", seed=42,
                task="PickCube-v1", use_goal=True, n_steps=10, guidance=1.0,
                execute_horizon=8, ema=False, n_episodes=100,
                success_rate=0.75, mean_length=137.0)
    return {**base, **kw}


def test_report_never_averages_across_observation_configs():
    """回归测试。use_goal 不进分组键时，PickCube 上接了 goal_pos 的 75% 与
    没接的 6% 会被聚合成 "29±40%" —— 一个不对应任何一次实验的数字，
    而且不会报错。这是本项目最贵的一课，钉死在这里。"""
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(ckpt="outputs/fmgoal_PickCube_s42/ckpt_20000.pt", use_goal=True,
             success_rate=0.75),
        _row(ckpt="outputs/fm_PickCube_s42/ckpt_20000.pt", use_goal=False,
             success_rate=0.06),
    ])
    out = mr.agg(mr.main_rows(mr.prepare(df)))
    assert len(out) == 2, f"两个观测配置被合并了：\n{out}"
    assert set(np.round(out["mean"], 2)) == {0.75, 0.06}


def test_report_keeps_each_method_at_its_own_default_operating_point():
    """主表要的是"各方法的默认工作点"，不是"所有步数的平均"。
    FM 10 步、DDPM 100 步、BC 一次前向。"""
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(model="fm", n_steps=10, success_rate=0.75),
        _row(model="fm", n_steps=1, success_rate=0.20),    # 步数消融，不该进主表
        _row(model="ddpm", ckpt="outputs/ddpmgoal_PickCube_s42/ckpt_20000.pt",
             n_steps=100, success_rate=0.60),
        _row(model="ddpm", ckpt="outputs/ddpmgoal_PickCube_s42/ckpt_20000.pt",
             n_steps=2, success_rate=0.05),                # 同上
    ])
    rows = mr.main_rows(mr.prepare(df))
    assert sorted(np.round(rows.success_rate, 2)) == [0.60, 0.75]


def test_report_excludes_non_default_execute_horizon():
    """执行长度是第二个旋钮，它的扫描结果同样不能混进主表。"""
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(execute_horizon=8, success_rate=0.75),
        _row(execute_horizon=1, success_rate=0.40),
    ])
    rows = mr.main_rows(mr.prepare(df))
    assert len(rows) == 1 and float(rows.success_rate.iloc[0]) == 0.75
