"""模型层单元测试。"""
import sys
from pathlib import Path

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
