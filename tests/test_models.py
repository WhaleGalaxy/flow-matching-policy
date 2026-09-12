"""模型层单元测试。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.denoiser import AdaLN, CrossAttention, FMDenoiser, SinusoidalPosEmb
from src.models.policy import (ActionNormalizer, EMA, fm_loss,
                               load_trainable_state_dict, trainable_state_dict)

D, H, A, L = 256, 16, 8, 83


def test_time_embedding_is_informative():
    """时间嵌入必须能区分不同的 t，否则 AdaLN 注入的条件等于噪声。"""
    emb = SinusoidalPosEmb(D)
    e = emb(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert e.shape == (5, D)
    sim = torch.nn.functional.cosine_similarity(e[0:1], e[-1:]).item()
    assert sim < 0.5, f"t=0 与 t=1 的嵌入过于相似 (cos={sim:.3f})"


def test_adaln_starts_as_identity():
    """零初始化：训练起步时 AdaLN 应等价于纯 LayerNorm。"""
    ada = AdaLN(D, D)
    x = torch.randn(4, H, D)
    out = ada(x, torch.randn(4, D))
    assert torch.allclose(out, torch.nn.functional.layer_norm(x, (D,)), atol=1e-5)


def test_cross_attention_shapes_and_context_dependence():
    ca = CrossAttention(D, D, n_heads=8)
    x = torch.randn(4, H, D)
    c1, c2 = torch.randn(4, L, D), torch.randn(4, L, D)
    assert ca(x, c1).shape == (4, H, D)
    # 输出投影是零初始化的，先给它非零权重再验证对 context 敏感
    torch.nn.init.normal_(ca.to_out.weight, std=0.02)
    assert not torch.allclose(ca(x, c1), ca(x, c2)), "输出与 context 无关，说明没接上"


def test_denoiser_zero_init_and_shapes():
    d = FMDenoiser(d_model=D, d_context=D, act_dim=A, act_horizon=H)
    out = d(torch.randn(4, H, A), torch.rand(4), torch.randn(4, L, D))
    assert out.shape == (4, H, A)
    assert out.abs().max() == 0, "输出层未零初始化，首步会预测非零速度"


def test_denoiser_rejects_wrong_horizon():
    d = FMDenoiser(d_model=D, d_context=D, act_dim=A, act_horizon=H)
    with pytest.raises(AssertionError):
        d(torch.randn(4, H + 1, A), torch.rand(4), torch.randn(4, L, D))


def test_fm_loss_positive_and_differentiable():
    d = FMDenoiser(d_model=D, d_context=D, act_dim=A, act_horizon=H)
    torch.nn.init.normal_(d.action_out.weight, std=0.02)  # 解除零初始化
    loss = fm_loss(d, torch.randn(4, H, A), torch.randn(4, L, D))
    assert loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.parameters())


def test_ema_tracks_but_lags_online_weights():
    d = FMDenoiser(d_model=D, d_context=D, act_dim=A, act_horizon=H)
    ema = EMA(d, decay=0.99)
    opt = torch.optim.SGD(d.parameters(), lr=0.1)
    before = ema.ema_model.action_in.weight.clone()
    sum(p.sum() for p in d.parameters()).backward()
    opt.step()
    ema.update(d)
    assert not torch.allclose(ema.ema_model.action_in.weight, before), "EMA 没更新"
    assert not torch.allclose(ema.ema_model.action_in.weight, d.action_in.weight), \
        "EMA 与在线权重相同，说明 decay 没生效"


def test_normalizer_roundtrip_and_persists_in_state_dict():
    """归一化统计量必须进 state_dict —— 推理时忘带 stats 是最常见的 bug。"""
    n = ActionNormalizer(A)
    actions = torch.randn(1000, H, A) * 3 + 5
    n.fit(actions)
    assert torch.allclose(n.denormalize(n.normalize(actions)), actions, atol=1e-4)
    assert n.normalize(actions).std().item() == pytest.approx(1.0, abs=0.05)

    restored = ActionNormalizer(A)
    restored.load_state_dict(n.state_dict())
    assert torch.allclose(restored.mean, n.mean) and bool(restored.fitted)


def test_action_representation_snr_detects_absolute_encoding():
    """信噪比检查必须能区分'绝对位置'和'增量'两种动作表示。

    这是 docs/debugging.md 里那个坑的回归测试：绝对表示下动作几乎完全由
    当前状态决定，任务信号趋近 0；增量表示下任务信号占主导。
    """
    import numpy as np
    from torch.utils.data import Dataset

    from src.diagnostics import action_representation_snr

    class Fake(Dataset):
        def __init__(self, absolute: bool, n=2048, H=16, A=7, P=9):
            g = torch.Generator().manual_seed(0)
            self.state = torch.randn(n, P, generator=g)
            delta = 0.1 * torch.randn(n, H, A, generator=g)   # 任务相关的动作
            base = self.state[:, :A].unsqueeze(1)              # 当前构型
            self.action = base + delta if absolute else delta
        def __len__(self): return len(self.action)
        def __getitem__(self, i):
            return {"rgb": torch.zeros(2, 3, 8, 8), "proprio": self.state[i].repeat(2, 1),
                    "action": self.action[i], "instruction": "x"}

    abs_snr = action_representation_snr(Fake(absolute=True), n_batches=4, num_workers=0)
    del_snr = action_representation_snr(Fake(absolute=False), n_batches=4, num_workers=0)
    assert abs_snr["signal"] < 0.15, f"绝对表示的任务信号应很低，得到 {abs_snr['signal']:.3f}"
    assert del_snr["signal"] > 0.80, f"增量表示的任务信号应很高，得到 {del_snr['signal']:.3f}"


def test_ema_checkpoint_roundtrip_keeps_weights():
    """EMA 权重必须真的存进 checkpoint。

    回归测试：EMA.__init__ 会把副本所有参数的 requires_grad 置为 False，而
    trainable_state_dict 正是用 requires_grad 判定"哪些不是冻结骨干"。在副本上
    这个判据恒为 False，曾导致存出的 ema 只剩 normalizer 的 3 个 buffer、
    没有任何权重 —— 训练中的评测照常（它直接用 EMA 对象），但事后所有
    从 checkpoint 做的评测和消融全部载入失败。
    """
    torch.manual_seed(0)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(4, 4)      # 冒充冻结骨干
            self.head = torch.nn.Linear(4, 2)        # 可训练
            self.register_buffer("stat", torch.ones(2))
            for p in self.frozen.parameters():
                p.requires_grad_(False)

    online = Toy()
    ema = EMA(online, decay=0.9)
    with torch.no_grad():
        online.head.weight.add_(1.0)
    ema.update(online)

    sd = trainable_state_dict(ema.ema_model, reference=online)
    assert "head.weight" in sd, f"EMA 权重丢失，只存下了 {sorted(sd)}"
    assert not any(k.startswith("frozen.") for k in sd), "冻结骨干不该被存进去"
    assert "stat" in sd, "buffer 必须保留"

    # 存下的键集合必须和在线模型一致，否则两边评测口径不同
    assert set(sd) == set(trainable_state_dict(online))

    # 能原样载回，且数值就是 EMA 的值而不是随机初始化
    fresh = Toy()
    load_trainable_state_dict(fresh, sd)
    torch.testing.assert_close(fresh.head.weight, ema.ema_model.head.weight)


def test_trainable_state_dict_refuses_to_save_nothing():
    """传错 reference 时必须响亮失败，而不是静默存出一个空 checkpoint。"""
    model = torch.nn.Linear(3, 3)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(AssertionError, match="没有任何可训练参数"):
        trainable_state_dict(model)


# ---------------------------------------------------------------- DDIM 采样器

def _zero_denoiser(x, t, ctx):
    """零初始化的 denoiser：FMDenoiser 的输出层就是零初始化的，所以刚建好的
    模型精确等价于这个函数。用它测采样器本身，不需要真权重。"""
    return torch.zeros_like(x)


def test_ddim_sampler_does_not_amplify_without_clipping_guard():
    """守一个让整个 DDPM 基线归零的 bug。

    cosine 调度 clamp 了 beta 上限，重新 cumprod 之后 `alphas_cumprod[-1] ≈ 2.4e-7`，
    比前一项小 1000 倍。采样恰好从这一项起步，`x0_pred = (x − …)/√ᾱ` 会把误差
    放大约 2029 倍。ε 预测为零时，整条链的净增益正好是 1/√ᾱ_last。

    症状是**所有配置下成功率完全相同**（实测采样步数 1→50、执行长度 1→16 全是
    2.0%），不会抛异常，只会安静地把基线变成常数输出。
    """
    from src.models.policy import DDPMPolicy, ddim_sample

    ac = DDPMPolicy._cosine_schedule(100)
    assert ac[-1] < ac[-2] / 100, "前提变了：调度末项不再是病态的小值，本测试需重写"

    x0 = torch.randn(4, H, 7)
    ctx = torch.zeros(4, L, D)

    unclipped = ddim_sample(_zero_denoiser, ac, x0.clone(), 100, 100, ctx, clip_x0=0)
    gain = unclipped.abs().max() / x0.abs().max()
    assert gain > 100, f"没有裁剪时本应炸开，实测增益只有 {gain:.1f}"

    clipped = ddim_sample(_zero_denoiser, ac, x0.clone(), 100, 100, ctx, clip_x0=4.0)
    assert torch.isfinite(clipped).all()
    assert clipped.abs().max() <= 4.0 + 1e-4, \
        f"裁剪后仍越界：{clipped.abs().max():.3f}"


@pytest.mark.parametrize("n_steps", [1, 2, 10, 100])
def test_ddim_sampler_bounded_at_every_step_count(n_steps):
    """少步是这个项目的主张所在，采样器在每个步数下都必须是有界的。"""
    from src.models.policy import DDPMPolicy, ddim_sample

    ac = DDPMPolicy._cosine_schedule(100)
    out = ddim_sample(_zero_denoiser, ac, torch.randn(4, H, 7), 100, n_steps,
                      torch.zeros(4, L, D), clip_x0=4.0)
    assert torch.isfinite(out).all() and out.abs().max() <= 4.0 + 1e-4


def test_checkpoint_roundtrip_preserves_non_default_architecture():
    """凡是进了模型构造函数的超参，load_policy 都必须从 cfg 读回来。

    实际踩过：BC 的等容量对照用 `hidden=2364`，而 load_policy 忘了读这一项，
    按默认 1024 建模型，于是所有 BC 评测都在 load_state_dict 处崩掉 ——
    而它们跑在后台脚本里，CSV 里只是**少了几行**，不看日志发现不了。
    """
    from src.models.policy import BCPolicy

    big = BCPolicy(hidden=333, d_model=D, act_dim=7, act_horizon=H, proprio_dim=9)
    state = trainable_state_dict(big)

    wrong = BCPolicy(hidden=1024, d_model=D, act_dim=7, act_horizon=H, proprio_dim=9)
    with pytest.raises(RuntimeError):
        load_trainable_state_dict(wrong, state)

    right = BCPolicy(hidden=333, d_model=D, act_dim=7, act_horizon=H, proprio_dim=9)
    load_trainable_state_dict(right, state)      # 不应抛异常


# ---------------------------------------------------------------------------
# 2026-09-11 加入的组件：等条件 BC 对照、OT 耦合、目标槽、冻结特征缓存
# ---------------------------------------------------------------------------

def test_bc_xattn_matches_fm_capacity():
    """等条件 BC 对照必须与 FM 同参数量，否则差距里又混进容量。

    只允许差一个 query（act_horizon × act_dim 个参数）—— 那是"没有噪声输入"
    这件事的全部代价。
    """
    from src.models.policy import BCXAttnPolicy, FMPolicy
    fm = FMPolicy(proprio_dim=13)
    bx = BCXAttnPolicy(proprio_dim=13)
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n(bx) - n(fm) == 16 * 7, (n(bx), n(fm))


def test_bc_xattn_forward_and_loss():
    from src.models.policy import BCXAttnPolicy
    p = BCXAttnPolicy(proprio_dim=13)
    p.normalizer.fit(torch.randn(64, 7))
    batch = dict(rgb=torch.randn(2, 2, 3, 126, 126), proprio=torch.randn(2, 2, 13),
                 action=torch.randn(2, 16, 7), instruction=["a", "b"])
    loss = p.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert p.predict_action(batch["rgb"], batch["proprio"], batch["instruction"]).shape \
        == (2, 16, 7)


def test_ot_coupling_is_a_permutation_and_lowers_cost():
    """minibatch OT 只能重排 x0，不能改动它的取值 —— 边缘分布必须原样保留，
    否则 CFM 回归的就不是同一个速度场了。"""
    from src.models.policy import ot_couple
    torch.manual_seed(0)
    x0 = torch.randn(32, 16, 7)
    x1 = torch.randn(32, 16, 7)
    out = ot_couple(x0, x1)
    before = (x0 - x1).pow(2).sum()
    after = (out - x1).pow(2).sum()
    assert after <= before
    # 排序后逐位相同 = 确实只是换了顺序
    assert torch.allclose(out.flatten().sort().values, x0.flatten().sort().values)


def test_logitnormal_time_concentrates_on_the_middle():
    """均匀采样把一半预算花在 t≈0 / t≈1，那两端的速度场近乎平凡。"""
    from src.models.policy import sample_flow_time
    torch.manual_seed(0)
    u = sample_flow_time(20000, "cpu", "uniform")
    ln = sample_flow_time(20000, "cpu", "logitnormal")
    mid = lambda t: ((t > 0.2) & (t < 0.8)).float().mean()
    assert 0.0 < ln.min() and ln.max() < 1.0
    assert mid(ln) > mid(u) + 0.15


def test_goal_slot_keeps_one_observation_space():
    """多任务的全部前提：有没有 goal_pos，proprio 维度都得一样。

    有效位不可省 —— 少了它，"目标在原点"与"这个任务没有目标"在网络看来一模一样。
    """
    from src.data.dataset import build_proprio
    q = np.random.randn(4, 9)
    with_goal = build_proprio(q, np.random.randn(4, 3), goal_slot=True)
    without = build_proprio(q, None, goal_slot=True)
    assert with_goal.shape == without.shape == (4, 13)
    assert torch.equal(with_goal[:, -1], torch.ones(4))
    assert torch.equal(without[:, -1], torch.zeros(4))
    assert torch.equal(without[:, 9:12], torch.zeros(4, 3))
    # 旧的 use_goal 行为不能被改坏（已发布的 12 维 checkpoint 还要能读）
    assert build_proprio(q, np.random.randn(4, 3)).shape == (4, 12)
    assert build_proprio(q).shape == (4, 9)


def test_visual_encoder_accepts_cached_tokens():
    """缓存路径与像素路径必须走到同一个 proj 上，且只靠维数分派。"""
    from src.models.encoders import VisualEncoder
    enc = VisualEncoder(d_model=32, img_size=126).eval()
    feats = torch.randn(2, 2, enc.n_patches, enc.embed_dim)
    with torch.no_grad():
        out = enc(feats)
    assert out.shape == (2, enc.n_patches, 32)
    with torch.no_grad():
        expect = enc.proj(feats.mean(dim=1))
    assert torch.allclose(out, expect)
