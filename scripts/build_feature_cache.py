"""把冻结 DINOv2 的 patch token 预先算好存盘。

为什么这是**数值等价**而不是近似：视觉骨干整个冻结（`VisualEncoder.freeze=True`，
参数 `requires_grad=False`，且始终 `eval()`），而数据管线里没有任何图像增强 ——
同一帧每个 epoch 喂进去的像素完全一样，算出来的 token 也就完全一样。可训练的
`proj` 层在缓存**之后**，所以它照常参与训练。

代价与收益：冻结骨干占约 88% 的训练时间，缓存后单轮 20000 步从约 40 分钟降到
约 8 分钟。3 个 seed、显著性检验、w/wo proprio 这些消融之所以一直没做，就是因为
一轮 40 分钟；这一步是让它们变得可做的前提。

存储布局（每个任务一个目录）：
    feat.f16       memmap (n_frames, n_patches, embed_dim) float16
    index.json     {episode_name: [offset, length]} + 用来校验的元信息

用法：
    python scripts/build_feature_cache.py --tasks PickCube-v1 PushCube-v1 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import default_h5_path, preprocess_obs_rgb  # noqa: E402
from src.models.encoders import DINOV2_MODEL, VisualEncoder  # noqa: E402


def episode_names(f: h5py.File, success_only: bool = True) -> list[str]:
    """必须与 ManiskillDataset 的扫描顺序和过滤条件完全一致，否则索引会错位。

    错位不会报错 —— 它会安静地把 A 集的图像配到 B 集的动作上，loss 照样下降，
    只是策略学到的是噪声。所以 index.json 里存了 episode 名字，训练时按名字取，
    而不是按序号。
    """
    names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
    out = []
    for name in names:
        ep = f[name]
        if success_only and not bool(np.asarray(ep["success"])[-1]):
            continue
        if ep["actions"].shape[0] < 1:
            continue
        out.append(name)
    return out


@torch.no_grad()
def build(task: str, out_root: Path, img_size: int = 126, camera: str = "base_camera",
          batch: int = 64, device: str = "cuda", source: str = "motionplanning") -> None:
    h5_path = default_h5_path(task, source=source)
    suffix = "" if source == "motionplanning" else f"_{source}"
    out_dir = out_root / f"{task}{suffix}_{camera}_{img_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = VisualEncoder(d_model=256, img_size=img_size).to(device).eval()
    n_patches, embed_dim = enc.n_patches, enc.embed_dim

    with h5py.File(h5_path, "r") as f:
        names = episode_names(f)
        # obs 比 actions 多一帧（最后一帧没有对应动作），缓存按 obs 的帧数存
        lengths = [f[n]["obs"]["sensor_data"][camera]["rgb"].shape[0] for n in names]
        total = int(sum(lengths))
        print(f"{task}: {len(names)} 条轨迹, {total:,} 帧 → "
              f"{total * n_patches * embed_dim * 2 / 1e9:.2f} GB", flush=True)

        feat_path = out_dir / "feat.f16"
        mm = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=np.float16, shape=(total, n_patches, embed_dim))

        index, off = {}, 0
        for i, (name, L) in enumerate(zip(names, lengths)):
            frames = np.asarray(f[name]["obs"]["sensor_data"][camera]["rgb"])
            for b0 in range(0, L, batch):
                chunk = frames[b0:b0 + batch]
                x = preprocess_obs_rgb(chunk, img_size).to(device)
                # VisualEncoder.forward 会把时间维平均掉，这里要的是逐帧特征，
                # 所以直接走骨干，绕开那一层平均
                feats = enc.dino.forward_features(x)[:, enc.dino.num_prefix_tokens:, :]
                mm[off + b0: off + b0 + len(chunk)] = feats.float().cpu().numpy().astype(np.float16)
            index[name] = [off, L]
            off += L
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(names)} 条", flush=True)
        mm.flush()
        del mm

    meta = {"task": task, "camera": camera, "img_size": img_size,
            "backbone": DINOV2_MODEL, "n_patches": n_patches, "embed_dim": embed_dim,
            "n_frames": total, "episodes": index}
    (out_dir / "index.json").write_text(json.dumps(meta))
    print(f"  已写出 {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path.home() / ".maniskill" / "feat_cache")
    ap.add_argument("--img-size", type=int, default=126)
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--source", default="motionplanning")
    args = ap.parse_args()
    for t in args.tasks:
        build(t, args.out, img_size=args.img_size, camera=args.camera,
              batch=args.batch, source=args.source)


if __name__ == "__main__":
    main()
