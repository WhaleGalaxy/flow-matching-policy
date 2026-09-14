"""分析与汇总代码的单元测试。

这些测试守的都是**安静出错**的地方：探针给出一个看似合理但没有意义的 R²、
汇总脚本把两个不同实验的数字平均成一个不对应任何实验的数字。
它们不会抛异常，只会让 README 里出现一个错的表。
"""
import importlib.util
import sys
import tempfile
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


def test_csv_migration_inserts_the_missing_column_in_place():
    """结果表加一列时，旧文件必须就地补齐，且补在**正确的位置**。

    补错位置不会报错：CSV 没有类型，错位之后 pandas 照样读出来，
    只是从此每一行的数字都对到了错误的列上。
    """
    import csv as _csv
    ru = _load("run_eval")
    header = ["ckpt", "task", "max_steps", "success_rate"]
    old = ["ckpt", "task", "success_rate"]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.csv"
        with open(f, "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(old); w.writerow(["a.pt", "PickCube-v1", "0.60"])
        ru.migrate_csv_header(f, header, migrations=(("max_steps", "300"),))
        rows = list(_csv.reader(open(f)))
    assert rows[0] == header
    assert rows[1] == ["a.pt", "PickCube-v1", "300", "0.60"], \
        f"补列补错了位置：{rows[1]}"


def test_csv_migration_refuses_a_table_it_cannot_repair():
    """少两列以上就该报错，而不是猜。混写两种 schema 是不会报错的那类错误。"""
    import csv as _csv
    ru = _load("run_eval")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.csv"
        with open(f, "w", newline="") as fh:
            _csv.writer(fh).writerow(["ckpt", "success_rate"])
        with pytest.raises(AssertionError):
            ru.migrate_csv_header(f, ["ckpt", "task", "max_steps", "success_rate"])


def test_report_never_averages_across_max_steps():
    """评测步数上限是成绩的一部分：300 步和 100 步不是同一个任务难度。

    ManiSkill 给 PickCube-v1 注册的上限是 50，官方基线用 100，本项目用 300。
    三者放进同一张表平均，得到的数字不对应任何一种评测协议。
    """
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(max_steps=300, success_rate=0.60),
        _row(max_steps=100, success_rate=0.35),
    ])
    out = mr.agg(mr.main_rows(mr.prepare(df)))
    assert len(out) == 2, f"两种评测预算被合并了：\n{out}"
    assert set(np.round(out["mean"], 2)) == {0.60, 0.35}


def test_report_disambiguates_same_seed_appearing_in_two_runs():
    """同一个 method 下同一个 seed 出现在两个 run 里，不可能是彼此的种子重复。

    旧 CSV 没有训练配置列，光靠表里的信息分不开它们 —— 实测
    outputs/2026-09-11/multitask.csv 的 8 个 fm run 就是这样被平均成 "4±2%" 的，
    而 README 里那一行是 60/96/36。分不开时要用 run 名兜底，不能安静地平均。
    """
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(ckpt="outputs/fmnocfg_mt_tg60_s42/ckpt_60000.pt", seed=42,
             success_rate=0.60),
        _row(ckpt="outputs/fmwide_mt_tg60_s42/ckpt_60000.pt", seed=42,
             success_rate=0.04),
    ])
    out = mr.agg(mr.main_rows(mr.prepare(df)))
    assert len(out) == 2, f"两个不同的 run 被当成种子重复平均了：\n{out}"
    assert set(np.round(out["mean"], 2)) == {0.60, 0.04}


def test_report_label_keeps_every_qualifier():
    """表里的名字必须能区分两行。丢掉限定词的话，4/95/8 和 60/96/36 会顶着
    同一个名字并排出现，读表的人无从分辨。"""
    mr = _load("make_report")
    a = mr.label_of("fm/multi/fmnocfg_mt_tg60_s42")
    b = mr.label_of("fm/multi/fmwide_mt_tg60_s42")
    assert a != b, f"两个不同的 run 显示成了同一个名字：{a}"
    assert "fmwide_mt_tg60_s42" in b
    assert mr.label_of("fm") == mr.LABEL["fm"], "已登记的名字不该被改写"


def test_report_counts_task_goal_as_a_goal_input():
    """task_goal 是第三种"给了目标"的观测配置，漏掉它会把多任务主线标成 nogoal。"""
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([_row(ckpt="outputs/fmnocfg_mt_tg60_s42/ckpt_60000.pt",
                            use_goal=False, goal_slot=False, task_goal=True)])
    assert "nogoal" not in mr.prepare(df).method.iloc[0]


def test_report_never_averages_across_n_average():
    """回归测试。n_average 是**另一个推理配置**，不是重复测量。

    run_eval.py 一直把它写进 CSV，但它从没进过 make_report 的分组键，于是同一个
    checkpoint 的 K=1/4/16 三行（实测 PickCube 60/56/53%）被当成同一配置的三次
    重复测量，平均成 56.3%。这与 use_goal 那一课是同一个形状：列记了，
    但汇总的时候没用上，而且不报错。
    """
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(n_average=1, success_rate=0.60),
        _row(n_average=4, success_rate=0.56),
        _row(n_average=16, success_rate=0.53),
    ])
    out = mr.agg(mr.main_rows(mr.prepare(df)))
    assert len(out) == 1, f"取均值的消融混进了主表：\n{out}"
    assert np.round(out["mean"].iloc[0], 2) == 0.60, \
        f"主表要的是推理时实际执行的单样本，拿到的却是 {out['mean'].iloc[0]:.3f}"


def test_report_never_averages_across_val_frac():
    """留出集比例不同 = 训练数据不同 = 不同的实验，不能当成两个 seed 平均。"""
    pd = pytest.importorskip("pandas")
    mr = _load("make_report")
    df = pd.DataFrame([
        _row(ckpt="outputs/fm_mt_s42/ckpt_60000.pt", val_frac=0.0, success_rate=0.60),
        _row(ckpt="outputs/fm_mt_v1_s42/ckpt_60000.pt", val_frac=0.1, success_rate=0.50),
    ])
    out = mr.agg(mr.main_rows(mr.prepare(df)))
    assert len(out) == 2, f"两种训练数据配置被合并了：\n{out}"
    assert set(np.round(out["mean"], 2)) == {0.60, 0.50}


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


def test_demo_composition_pads_short_panels_and_keeps_frame_count():
    """动图合成：三个任务的长度不同，短的要用最后一帧补齐。

    不补齐的话先结束的任务要么凭空消失（形状对不上直接崩），要么循环重放 ——
    后者看起来像策略在反复尝试，把一次成功讲成了一次挣扎。
    """
    rd = _load("record_demo")
    panels = [np.zeros((30, 64, 64, 3), np.uint8),
              np.full((12, 64, 64, 3), 255, np.uint8),
              np.zeros((7, 64, 64, 3), np.uint8)]
    anim = rd.compose(panels, ["A", "B", "C"], scale=32, stride=3)
    assert anim.shape[1] == 32 and anim.shape[2] == 32 * 3
    idx = list(range(0, 30, 3))
    expect = len(idx) + (0 if idx[-1] == 29 else 1)   # 末帧一定收进来
    assert anim.shape[0] == expect, "抽帧数量不对"
    # 最后一帧里，B 那一格应当还是它自己的最后一帧（白），不是黑或越界
    assert anim[-1, 16, 32 + 16].max() > 200, "短序列没有被正确补齐"
