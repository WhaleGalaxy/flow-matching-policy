"""Training entry point, driven by Hydra config (configs/train.yaml).

Usage (once implemented):
    python -m src.train task=pick model=fm seed=42
    python -m src.train ++lr=3e-4 ++task=peg ++seed=123

Checklist ids: training loop itself is built incrementally through Week 1
(w1d2t1 for BC, w1d4t1 for FM); this file is where it's wired into Hydra +
WandB during Week 2 Day 4 (w2d4t1).
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "w1d2t1 / w1d4t1 / w2d4t1: build dataset+dataloader, model, optimizer, "
        "WandB logging, and the train loop here."
    )


if __name__ == "__main__":
    main()
