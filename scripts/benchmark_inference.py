"""实测推理延迟：FM vs DDPM，不同采样步数，含/不含 CFG。

延迟只取决于架构和采样步数，与训练质量无关，所以用随机初始化的权重测即可。
两个模型骨干完全相同，差异纯粹来自采样步数需求 —— 这正是要量化的东西。

注意：测的是**单次动作块预测**的延迟。实际控制频率还要除以 execute_horizon
（一次预测执行 8 步），所以有效控制频率是这个数字的 8 倍。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.policy import DDPMPolicy, FMPolicy  # noqa: E402

DEV = "cuda"
BATCH = 1          # 部署时是单个机器人，batch=1 才是真实场景
EXECUTE_HORIZON = 8


@torch.no_grad()
def timeit(policy, n_steps, guidance, iters=50, warmup=10):
    rgb = torch.randn(BATCH, 2, 3, 126, 126, device=DEV)
    prop = torch.randn(BATCH, 2, 9, device=DEV)
    inst = ["Pick up the red cube"] * BATCH

    for _ in range(warmup):
        policy.predict_action(rgb, prop, inst, n_steps=n_steps, guidance=guidance)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        policy.predict_action(rgb, prop, inst, n_steps=n_steps, guidance=guidance)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}   batch={BATCH}\n")

    common = dict(act_dim=7, act_horizon=16, img_size=126, proprio_dim=9)
    fm = FMPolicy(**common).to(DEV).eval()
    ddpm = DDPMPolicy(**common).to(DEV).eval()

    # 观测编码的固定开销（两个模型完全一样），单独测出来便于归因
    rgb = torch.randn(BATCH, 2, 3, 126, 126, device=DEV)
    prop = torch.randn(BATCH, 2, 9, device=DEV)
    with torch.no_grad():
        for _ in range(10):
            fm.encode_obs(rgb, prop, ["x"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fm.encode_obs(rgb, prop, ["x"])
        torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) / 50 * 1e3
    print(f"观测编码固定开销（DINOv2+SigLIP+proprio）: {enc_ms:.2f} ms —— 两模型共有\n")

    rows = []
    print(f"{'模型':<8}{'采样步数':>9}{'CFG':>6}{'延迟(ms)':>11}{'去噪部分':>11}{'有效控制频率':>14}")
    print("-" * 62)
    for name, policy, steps in (("FM", fm, [1, 2, 5, 10, 20]),
                                ("DDPM", ddpm, [10, 50, 100])):
        for s in steps:
            for g in (1.0, 1.5):
                ms = timeit(policy, s, g)
                hz = 1000.0 / ms * EXECUTE_HORIZON
                rows.append((name, s, g, ms))
                print(f"{name:<8}{s:>9}{('无' if g==1.0 else f'w={g}'):>6}"
                      f"{ms:>11.2f}{ms-enc_ms:>11.2f}{hz:>12.0f} Hz")
    print("-" * 62)

    fm10 = next(r[3] for r in rows if r[0] == "FM" and r[1] == 10 and r[2] == 1.0)
    dd100 = next(r[3] for r in rows if r[0] == "DDPM" and r[1] == 100 and r[2] == 1.0)
    print(f"\nFM(10步) {fm10:.1f} ms  vs  DDPM(100步) {dd100:.1f} ms  →  {dd100/fm10:.1f}x 更快")
    print(f"扣掉共有的编码开销后，纯去噪部分快 {(dd100-enc_ms)/(fm10-enc_ms):.1f}x")
    print("\n注：DDPM 用 100 步是 Diffusion Policy 的常规设置；两者是否在相同步数下")
    print("    达到相同成功率，由 notebooks 里的采样步数消融实验回答。")


if __name__ == "__main__":
    main()
