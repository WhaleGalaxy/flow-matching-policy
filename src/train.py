"""训练入口（Hydra 驱动）。

    python -m src.train                                   # 默认 FM + PickCube
    python -m src.train model=bc                          # BC 基线
    python -m src.train tasks="[PickCube-v1,StackCube-v1,PegInsertionSide-v1]"
    python -m src.train ++lr=3e-4 ++seed=123
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import (ManiskillDataset, build_multitask_dataset,  # noqa: E402
                              collate, default_h5_path)
from src.evaluate import make_env, rollout  # noqa: E402
from src.models.policy import BCPolicy, DDPMPolicy, EMA, FMPolicy  # noqa: E402


def build_dataset(cfg):
    kw = dict(obs_horizon=cfg.obs_horizon, act_horizon=cfg.act_horizon,
              img_size=cfg.img_size, max_episodes=cfg.max_episodes)
    tasks = list(cfg.tasks)
    if len(tasks) == 1:
        ds = ManiskillDataset(default_h5_path(tasks[0]), task=tasks[0], **kw)
        return ds, None, [ds]
    return build_multitask_dataset(tasks, **kw)


def build_policy(cfg, act_dim, proprio_dim):
    common = dict(act_dim=act_dim, act_horizon=cfg.act_horizon, img_size=cfg.img_size,
                  proprio_dim=proprio_dim, use_proprio=cfg.use_proprio,
                  d_model=cfg.model.d_model)
    if cfg.model.name == "bc":
        return BCPolicy(**common)
    gen = dict(n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
               cfg_dropout=cfg.model.get("cfg_dropout", 0.2), **common)
    if cfg.model.name == "ddpm":
        return DDPMPolicy(n_train_steps=cfg.model.n_train_steps, **gen)
    return FMPolicy(**gen)


def lr_at(step, cfg):
    """线性预热 + 余弦退火。"""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True   # 实测 1.3-2.9x，白拿的加速
    torch.backends.cudnn.allow_tf32 = True

    run_name = cfg.run_name or f"{cfg.model.name}_{'+'.join(t.split('-')[0] for t in cfg.tasks)}_s{cfg.seed}"
    out_dir = Path(hydra.utils.get_original_cwd()) / cfg.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    print(f"输出目录: {out_dir}\n")

    # ---- 数据 ----
    dataset, sampler, subsets = build_dataset(cfg)
    print(f"数据集: {len(dataset):,} 个样本 / {len(cfg.tasks)} 个任务")
    for d in subsets:
        print(f"  {d.task:22s} {len(d):>8,} 样本  act_dim={d.act_dim} proprio_dim={d.proprio_dim}")

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )

    # ---- 模型 ----
    policy = build_policy(cfg, subsets[0].act_dim, subsets[0].proprio_dim).to(cfg.device)
    # 归一化统计量从训练集拟合，存进 buffer 随 checkpoint 一起走
    all_actions = torch.cat([d.sample_actions() for d in subsets])
    policy.normalizer.fit(all_actions.to(cfg.device))
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\n模型 {cfg.model.name}: {n_train/1e6:.2f}M 可训练参数")

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(policy, decay=cfg.model.get("ema_decay", 0.9999))

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb.project, name=run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # ---- 训练 ----
    policy.train()
    step, t0, running = 0, time.perf_counter(), []
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break
            for k in ("rgb", "proprio", "action"):
                batch[k] = batch[k].to(cfg.device, non_blocking=True)

            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                loss = policy.compute_loss(batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ema.update(policy)

            running.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg = sum(running) / len(running); running = []
                ips = step / (time.perf_counter() - t0)
                print(f"step {step:6d}/{cfg.steps}  loss {avg:.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  |g| {gn:.2f}  {ips:.1f} it/s")
                if use_wandb:
                    wandb.log({"train/loss": avg, "train/lr": opt.param_groups[0]["lr"],
                               "train/grad_norm": float(gn), "step": step})

            if step % cfg.eval.every == 0 or step == cfg.steps:
                ckpt = out_dir / f"ckpt_{step}.pt"
                torch.save({"step": step, "cfg": OmegaConf.to_container(cfg, resolve=True),
                            "model": policy.state_dict(),
                            "ema": ema.ema_model.state_dict()}, ckpt)
                print(f"  已保存 {ckpt.name}")

                if cfg.eval.n_episodes > 0:
                    # 评测一律用 EMA 权重
                    scores = {}
                    for task in cfg.tasks:
                        env = make_env(task)
                        try:
                            r = rollout(ema.ema_model, env, task,
                                        n_episodes=cfg.eval.n_episodes,
                                        obs_horizon=cfg.obs_horizon,
                                        execute_horizon=cfg.eval.execute_horizon,
                                        img_size=cfg.img_size,
                                        n_steps=cfg.model.get("n_sample_steps", 10),
                                        guidance=cfg.eval.get("guidance", 1.0))
                        finally:
                            env.close()
                        scores[task] = r["success_rate"]
                        print(f"  [eval] {task:22s} SR {100*r['success_rate']:5.1f}%  "
                              f"平均步长 {r['mean_length']:.0f}")
                    if use_wandb:
                        wandb.log({f"eval/{t}_sr": v for t, v in scores.items()} | {"step": step})
                    policy.train()

    print(f"\n训练完成，用时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
