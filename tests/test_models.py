"""模型层单元测试。"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.denoiser import AdaLN, CrossAttention, FMDenoiser, SinusoidalPosEmb
from src.models.policy import ActionNormalizer, EMA, fm_loss

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
