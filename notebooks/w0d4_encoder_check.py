"""Week0 Day4：验证三个编码器 + 在本机 8GB 卡上实测显存与吞吐。

要回答的问题：
  1. DINOv2 输入分辨率取多少？（patch=14 的约束下，分辨率直接决定 token 数和开销）
  2. batch=128 + 两帧观测，8GB 显存够不够？
  3. SigLIP 对三个任务指令的编码是否可区分？（否则语言条件形同虚设）
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoders import LanguageEncoder, ProprioEncoder, VisualEncoder  # noqa: E402

DEV = "cuda"
INSTRUCTIONS = [
    "Pick up the red cube",
    "Stack cube A on cube B",
    "Insert the peg into the hole",
]


def probe_visual(img_size, batch=128, frames=2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    enc = VisualEncoder(img_size=img_size).to(DEV).eval()

    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in enc.parameters() if not p.requires_grad)

    imgs = torch.randn(batch, frames, 3, img_size, img_size, device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            out = enc(imgs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = enc(imgs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10

    peak = torch.cuda.max_memory_allocated() / 1e9
    del enc, imgs, out
    torch.cuda.empty_cache()
    return dict(patches=(img_size // 14) ** 2, ms=dt * 1e3, mem=peak,
                train=n_train, frozen=n_frozen)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("=== 1. DINOv2 分辨率选型 (batch=128, 2帧) ===")
    print(f"{'img_size':>9} {'patches':>8} {'前向(ms)':>10} {'峰值显存(GB)':>13}")
    for size in (98, 126, 154, 224):
        r = probe_visual(size)
        print(f"{size:>9} {r['patches']:>8} {r['ms']:>10.1f} {r['mem']:>13.2f}")
    print(f"\n可训练参数 {r['train']:,} / 冻结参数 {r['frozen']:,}"
          f"  → 冻结比例 {100*r['frozen']/(r['train']+r['frozen']):.1f}%")

    print("\n=== 2. 冻结是否真的生效 ===")
    enc = VisualEncoder(img_size=126).to(DEV)
    enc.train()  # 即使调用 train()，骨干也应保持 eval
    assert not enc.dino.training, "冻结的骨干不该进入 train 模式"
    assert sum(p.requires_grad for p in enc.dino.parameters()) == 0, "骨干仍有梯度"
    print("    OK — 骨干 requires_grad 全为 False，且 train() 后仍在 eval 模式")

    print("\n=== 3. SigLIP 语言编码可区分性 ===")
    lang = LanguageEncoder().to(DEV).eval()
    with torch.no_grad():
        raw = lang._encode(INSTRUCTIONS + [""])           # 投影前的原始 768 维
        tok = lang(INSTRUCTIONS)                           # 投影后的 token
    print(f"    输出 token shape: {tuple(tok.shape)}")
    sim = F.normalize(raw, dim=-1) @ F.normalize(raw, dim=-1).T
    names = ["Pick", "Stack", "Peg", "<null>"]
    print("    余弦相似度矩阵：")
    print("           " + "".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"    {n:>7}" + "".join(f"{sim[i,j]:>9.3f}" for j in range(len(names))))
    off_diag = sim[:3, :3][~torch.eye(3, dtype=bool, device=DEV)]
    print(f"    三任务两两相似度 max={off_diag.max():.3f} "
          f"(远低于 1.0 才说明语言条件有信息量)")

    print("\n=== 4. 指令缓存 ===")
    lang.clear_cache()
    t0 = time.perf_counter(); lang(INSTRUCTIONS * 40); t_cold = time.perf_counter() - t0
    t0 = time.perf_counter(); lang(INSTRUCTIONS * 40); t_warm = time.perf_counter() - t0
    print(f"    首次 {t_cold*1e3:.1f} ms → 命中缓存 {t_warm*1e3:.1f} ms "
          f"({t_cold/max(t_warm,1e-9):.0f}x)，缓存了 {len(lang._cache)} 条指令")

    print("\n=== 5. 本体感知编码器 ===")
    prop = ProprioEncoder().to(DEV)
    out = prop(torch.randn(128, 2, 9, device=DEV))
    print(f"    (128,2,9) -> {tuple(out.shape)}")

    print("\n=== 6. 三编码器同时驻留的显存 ===")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    vis = VisualEncoder(img_size=126).to(DEV).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v = vis(torch.randn(128, 2, 3, 126, 126, device=DEV))
        l = lang(INSTRUCTIONS * 42 + INSTRUCTIONS[:2])
        p = prop(torch.randn(128, 2, 9, device=DEV))
        context = torch.cat([v, l.to(v.dtype), p.to(v.dtype)], dim=1)
    print(f"    context shape: {tuple(context.shape)}  "
          f"(= {v.shape[1]} 视觉 + 1 语言 + 1 本体)")
    print(f"    峰值显存 {torch.cuda.max_memory_allocated()/1e9:.2f} GB / 8.2 GB")


if __name__ == "__main__":
    main()
