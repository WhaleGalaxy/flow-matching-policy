"""Unit tests for shapes / basic invariants. Checklist id: w2d4t2.

Each test is skipped until the corresponding module is implemented -- unskip
it (remove the `pytest.mark.skip` line) as you finish that day's task.
"""
import pytest
import torch


@pytest.mark.skip(reason="unskip after w1d0t0 (VisualEncoder)")
def test_encoder_shapes():
    from src.models.encoders import VisualEncoder

    enc = VisualEncoder()
    imgs = torch.randn(2, 2, 3, 96, 96)
    out = enc(imgs)
    assert out.shape == (2, 256, 256)  # (B, N_patches, d_model)


@pytest.mark.skip(reason="unskip after w1d3t1 + w1d4t0 (FMDenoiser + fm_loss)")
def test_fm_loss_positive():
    from src.models.policy import FMPolicy, fm_loss

    policy = FMPolicy()
    action = torch.randn(4, 16, 7)
    context = torch.randn(4, 50, 256)
    loss = fm_loss(policy.denoiser, action, context)
    assert loss.item() > 0


@pytest.mark.skip(reason="unskip after w1d4t0 (EMA)")
def test_ema_weights_differ():
    from src.models.policy import EMA, FMPolicy

    policy = FMPolicy()
    ema = EMA(policy.denoiser)
    optimizer = torch.optim.Adam(policy.denoiser.parameters(), lr=1e-3)
    dummy_loss = sum(p.sum() for p in policy.denoiser.parameters())
    dummy_loss.backward()
    optimizer.step()
    ema.update(policy.denoiser)
    for ep, sp in zip(ema.ema_model.parameters(), policy.denoiser.parameters()):
        assert not torch.allclose(ep.data, sp.data)
