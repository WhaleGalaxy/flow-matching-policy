"""训练入口（Hydra 驱动）。

注意：日志重定向到文件时 stdout 是块缓冲的，所有 print 都带 flush=True，
否则后台训练看不到实时进度。

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
from src.data.dataset import (ManiskillDataset, MixedSourceDataset,  # noqa: E402
                              build_multitask_dataset, collate, default_h5_path)
from src.diagnostics import check_dataset  # noqa: E402
from src.evaluate import make_env, rollout  # noqa: E402
from src.models.policy import (BCPolicy, BCXAttnPolicy, DDPMPolicy, EMA,  # noqa: E402
                               FMPolicy, load_trainable_state_dict,
                               trainable_state_dict)


def build_dataset(cfg):
    kw = dict(obs_horizon=cfg.obs_horizon, act_horizon=cfg.act_horizon,
              img_size=cfg.img_size, max_episodes=cfg.max_episodes,
              use_goal=cfg.get("use_goal", False),
              goal_slot=cfg.get("goal_slot", False),
              task_goal=cfg.get("task_goal", False),
              feature_cache=cfg.get("feature_cache", None))
    tasks = list(cfg.tasks)
    sources = list(cfg.get("sources", ["motionplanning"]))
    if len(sources) > 1:
        # 同一个任务的多份演示（不同解法）混合。存在的理由见
        # src/data/dataset.default_h5_path：单一来源的条件动作分布是宽的单峰，
        # 生成式动作头没有多模态可表示；混合来源才造得出真正的多模态。
        assert len(tasks) == 1, "混合演示来源目前只支持单任务"
        ds = MixedSourceDataset(tasks[0], sources, **kw)
        return ds, None, [ds]
    if len(tasks) == 1:
        ds = ManiskillDataset(default_h5_path(tasks[0], source=sources[0]),
                              task=tasks[0], **kw)
        return ds, None, [ds]
    return build_multitask_dataset(tasks, **kw)


def build_policy(cfg, act_dim, proprio_dim):
    common = dict(act_dim=act_dim, act_horizon=cfg.act_horizon, img_size=cfg.img_size,
                  proprio_dim=proprio_dim, use_proprio=cfg.use_proprio,
                  d_model=cfg.model.d_model)
    if cfg.model.name == "bc":
        return BCPolicy(hidden=cfg.model.get("hidden", 1024), **common)
    gen = dict(n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
               cfg_dropout=cfg.model.get("cfg_dropout", 0.2),
               global_cond=cfg.get("global_cond", False), **common)
    if cfg.model.name == "bc_xattn":
        return BCXAttnPolicy(**gen)
    if cfg.model.name == "ddpm":
        return DDPMPolicy(n_train_steps=cfg.model.n_train_steps, **gen)
    return FMPolicy(ot_coupling=cfg.model.get("ot_coupling", False),
                    time_dist=cfg.model.get("time_dist", "uniform"),
                    time_loc=cfg.model.get("time_loc", 0.0),
                    time_scale=cfg.model.get("time_scale", 1.0), **gen)


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
    print(OmegaConf.to_yaml(cfg), flush=True)
    print(f"输出目录: {out_dir}\n", flush=True)

    # ---- 数据 ----
    dataset, sampler, subsets = build_dataset(cfg)
    print(f"数据集: {len(dataset):,} 个样本 / {len(cfg.tasks)} 个任务", flush=True)
    for d in subsets:
        print(f"  {d.task:22s} {len(d):>8,} 样本  act_dim={d.act_dim} proprio_dim={d.proprio_dim}", flush=True)

    # 训练前健全性检查 —— 几秒钟，但能拦住整类"loss 很低却完全不工作"的问题
    print("\n训练前检查：", flush=True)
    check_dataset(subsets[0])
    print("", flush=True)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.num_workers > 0, pin_memory=True,
    )

    # ---- 模型 ----
    act_dim, proprio_dim = subsets[0].act_dim, subsets[0].proprio_dim
    for d in subsets[1:]:
        assert (d.act_dim, d.proprio_dim) == (act_dim, proprio_dim), \
            f"任务间的动作/本体维度不一致：{d.task}"
    OmegaConf.set_struct(cfg, False)
    cfg.act_dim, cfg.proprio_dim = act_dim, proprio_dim   # 存进 ckpt，评测时直接读
    OmegaConf.set_struct(cfg, True)
    policy = build_policy(cfg, act_dim, proprio_dim).to(cfg.device)
    # 归一化统计量从训练集拟合，存进 buffer 随 checkpoint 一起走
    all_actions = torch.cat([d.sample_actions() for d in subsets])
    policy.normalizer.fit(all_actions.to(cfg.device))
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\n模型 {cfg.model.name}: {n_train/1e6:.2f}M 可训练参数", flush=True)

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(policy, decay=cfg.model.get("ema_decay", 0.9999))

    # ---- 断点续跑 ----
    # 9-10 那一轮是机器关机丢掉的，损失是整轮训练。没有续跑机制的话，
    # 每次意外都要从第 0 步重来。last.pt 里除权重外还有优化器状态和步数：
    # 少了优化器动量，续跑等于把 AdamW 重新预热一遍，loss 会有个可见的台阶。
    start_step = 0
    last_path = out_dir / "last.pt"
    if cfg.get("resume", True) and last_path.exists():
        blob = torch.load(last_path, map_location=cfg.device, weights_only=False)
        load_trainable_state_dict(policy, blob["model"])
        load_trainable_state_dict(ema.ema_model, blob["ema"])
        opt.load_state_dict(blob["opt"])
        ema.step = blob.get("ema_step", blob["step"])
        start_step = blob["step"]
        print(f"\n从 {last_path.name} 续跑：step {start_step}/{cfg.steps}", flush=True)
        if start_step >= cfg.steps:
            print("已经跑完，无需继续。", flush=True)
            return

    def save_last(step):
        tmp = last_path.with_suffix(".tmp")
        torch.save({"step": step, "ema_step": ema.step,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "model": trainable_state_dict(policy),
                    "ema": trainable_state_dict(ema.ema_model, reference=policy),
                    "opt": opt.state_dict()}, tmp)
        tmp.replace(last_path)   # 原子替换：写一半时断电也不会毁掉上一个可用的

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"),
                   name=run_name, config=OmegaConf.to_container(cfg, resolve=True),
                   tags=[cfg.model.name, *cfg.tasks, *cfg.wandb.get("tags", [])])
        # loss 只能看训练崩没崩，验收看成功率 —— 让网页默认按成功率排 run
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step", summary="max")

    # ---- 训练 ----
    policy.train()
    step, t0, running = start_step, time.perf_counter(), []
    steps_done_this_run = 0
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
            steps_done_this_run += 1

            if step % 100 == 0:
                avg = sum(running) / len(running); running = []
                ips = steps_done_this_run / (time.perf_counter() - t0)
                print(f"step {step:6d}/{cfg.steps}  loss {avg:.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  |g| {gn:.2f}  {ips:.1f} it/s", flush=True)
                if use_wandb:
                    wandb.log({"train/loss": avg, "train/lr": opt.param_groups[0]["lr"],
                               "train/grad_norm": float(gn), "step": step})

            if step % cfg.get("ckpt_every", 1000) == 0 and step % cfg.eval.every != 0:
                save_last(step)

            if step % cfg.eval.every == 0 or step == cfg.steps:
                ckpt = out_dir / f"ckpt_{step}.pt"
                torch.save({"step": step, "cfg": OmegaConf.to_container(cfg, resolve=True),
                            "model": trainable_state_dict(policy),
                            "ema": trainable_state_dict(ema.ema_model, reference=policy)}, ckpt)
                save_last(step)
                print(f"  已保存 {ckpt.name}", flush=True)

                if cfg.eval.n_episodes > 0:
                    # 用哪套权重不能照抄惯例。本项目实测 decay=0.9999 的时间常数
                    # (10000 步) 对 20000 步的训练太慢：同一批 checkpoint，在线权重
                    # 是 6/37/59/70/75%，EMA 读数却是平的 8/4/4/4/4% —— 按后者会
                    # 得出"训练毫无进展"的相反结论。默认用在线权重，训练更久时
                    # 可以用 ++eval.use_ema=true 换回来。
                    use_ema = cfg.eval.get("use_ema", False)
                    eval_model = ema.ema_model if use_ema else policy
                    eval_model.eval()
                    scores = {}
                    for task in cfg.tasks:
                        env = make_env(task)
                        try:
                            r = rollout(eval_model, env, task,
                                        n_episodes=cfg.eval.n_episodes,
                                        obs_horizon=cfg.obs_horizon,
                                        execute_horizon=cfg.eval.execute_horizon,
                                        img_size=cfg.img_size,
                                        n_steps=cfg.model.get("n_sample_steps", 10),
                                        guidance=cfg.eval.get("guidance", 1.0),
                                        use_goal=cfg.get("use_goal", False),
                                        goal_slot=cfg.get("goal_slot", False),
                                        task_goal=cfg.get("task_goal", False))
                        finally:
                            env.close()
                        scores[task] = r["success_rate"]
                        print(f"  [eval] {task:22s} SR {100*r['success_rate']:5.1f}%  "
                              f"平均步长 {r['mean_length']:.0f}  "
                              f"({'EMA' if use_ema else '在线'}权重)", flush=True)
                    if use_wandb:
                        wandb.log({f"eval/{t}_sr": v for t, v in scores.items()} | {"step": step})
                    policy.train()

    print(f"\n训练完成，用时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
