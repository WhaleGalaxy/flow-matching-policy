"""Week0 Day0：在本机 RTX 4060 Laptop 上实测 FP32 / TF32 / BF16 的训练吞吐。

三档精度的区别：
  fp32 : 严格 FP32 矩阵乘（torch 2.x 默认，matmul 的 TF32 是关的）
  tf32 : Ampere+ 的 TensorFloat-32，19-bit 尾数，一行开关，无需改代码
  bf16 : autocast 自动混合精度。注意 **不需要 GradScaler**（BF16 指数位=FP32，不会下溢）

用两个模型测：ResNet-18（卷积密集，对应计划里的 CIFAR-10）和一个尺寸与本项目
FMDenoiser 一致的 Transformer（这才是我们真正要训的东西）。
"""
import time

import torch
import torch.nn as nn
import torchvision

DEV = "cuda"


def bench(build_model, make_batch, mode, iters=40, warmup=10):
    """返回 (iters/s, 峰值显存GB)。mode ∈ {fp32, tf32, bf16}"""
    torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
    torch.backends.cudnn.allow_tf32 = (mode == "tf32")

    torch.manual_seed(0)
    model = build_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.MSELoss()
    torch.cuda.reset_peak_memory_stats()

    def step():
        x, target = make_batch()
        # BF16 只需要 autocast，不需要 GradScaler
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "bf16")):
            out = model(x)
            loss = crit(out.float(), target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return iters / dt, peak


class DenoiserLike(nn.Module):
    """尺寸与本项目 FMDenoiser 一致：d_model=256, 4层, 8头, 动作序列16 + context 50。"""

    def __init__(self, d_model=256, n_layers=4, n_heads=8, horizon=16, ctx_len=50):
        super().__init__()
        self.horizon, self.ctx_len, self.d = horizon, ctx_len, d_model
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.net = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d_model, 7)

    def forward(self, x):
        ctx = x[:, self.horizon:, :]
        act = x[:, : self.horizon, :]
        return self.out(self.net(act, ctx))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)\n")

    B = 128
    suites = {
        "ResNet-18 (B=128, 3x32x32)": (
            lambda: torchvision.models.resnet18(num_classes=10),
            lambda: (torch.randn(B, 3, 32, 32, device=DEV),
                     torch.randn(B, 10, device=DEV)),
        ),
        "FMDenoiser-like Transformer (B=128)": (
            DenoiserLike,
            lambda: (torch.randn(B, 16 + 50, 256, device=DEV),
                     torch.randn(B, 16, 7, device=DEV)),
        ),
    }

    for name, (build, batch) in suites.items():
        print(f"--- {name} ---")
        base = None
        for mode in ("fp32", "tf32", "bf16"):
            ips, peak = bench(build, batch, mode)
            base = base or ips
            print(f"  {mode:4s}  {ips:7.1f} it/s   {ips/base:4.2f}x   峰值显存 {peak:.2f} GB")
        print()


if __name__ == "__main__":
    main()
