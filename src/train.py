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


def build_dataset(cfg, split: str = "train"):
    kw = dict(obs_horizon=cfg.obs_horizon, act_horizon=cfg.act_horizon,
              img_size=cfg.img_size, max_episodes=cfg.max_episodes,
              use_goal=cfg.get("use_goal", False),
              goal_slot=cfg.get("goal_slot", False),
              task_goal=cfg.get("task_goal", False),
              feature_cache=cfg.get("feature_cache", None),
              val_frac=cfg.get("val_frac", 0.0),
              split_seed=cfg.get("split_seed", 0), split=split)
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


def diverging(vl: float, best: float, bad: int, step: int, total: int,
              ratio: float = 1.5, warmup_frac: float = 0.2) -> int:
    """更新"验证 loss 连续变坏了几次"的计数。返回新的计数，>=2 即判为发散。

    为什么用验证 loss 而不是训练 loss：训练 loss 被 batch 噪声盖着，涨起来不明显；
    验证 loss 固定了数据与 RNG，是一条干净的曲线。2026-09-12 那次发散里，
    训练 loss 从 0.144 涨到 0.687 花了两万五千步才显眼，而验证 loss 在
    0.158 → 0.277 那一步就已经翻了 1.76 倍。

    为什么要连续两次：单次抖动不该判死刑。
    为什么前 20% 不生效：早期 loss 本来就在剧烈下降，最低点还没稳定下来。
    """
    if step <= warmup_frac * total:
        return 0
    return bad + 1 if vl > ratio * best else 0


@torch.no_grad()
def val_loss(policy, loader, cfg, gen, n_batches: int) -> float:
    """留出集上的 loss。

    **RNG 必须固定住。** FM 和 DDPM 的 loss 每算一次都要现采流时间 t 和噪声 x0，
    直接算出来的读数带着这层采样噪声，step 之间不可比 —— 曲线的抖动会盖过真正的
    变化，而这正是这条曲线要看的东西（加容量之后 val loss 有没有掉头向上）。
    所以每次都用同一批数据、同一组 t、同一组 x0，差异就只来自模型本身。

    算完把全局 RNG 状态还回去：val 消耗掉的随机数如果不还，训练侧的数据增广、
    CFG dropout 序列就会随"这一步有没有算 val"而改变，实验不再可复现。
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(12345)
    gen.manual_seed(12345)          # 让每次取到的是**同一批** val 样本
    was_training = policy.training
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        for k in ("rgb", "proprio", "action"):
            batch[k] = batch[k].to(cfg.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
            total += float(policy.compute_loss(batch))
        n += 1
    if was_training:
        policy.train()
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return total / max(1, n)


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

    # ---- 留出集 ----
    # 划分由 split_seed 决定而非 cfg.seed，所以同一批对照实验（FM / DDPM / BC）
    # 拿到的是逐条相同的训练集与留出集。见 src/data/dataset.split_episodes。
    val_loader, val_gen = None, None
    if cfg.get("val_frac", 0.0) > 0:
        val_dataset, _, val_subsets = build_dataset(cfg, split="val")
        n_val_ep = sum(len(d.episode_lengths) for d in val_subsets)
        n_tr_ep = sum(len(d.episode_lengths) for d in subsets)
        print(f"留出集: {len(val_dataset):,} 个样本 / {n_val_ep} 条轨迹"
              f"（训练 {n_tr_ep} 条，val_frac={cfg.val_frac} split_seed={cfg.split_seed}）",
              flush=True)
        # 逐条核对训练与留出没有交集。断言而不是注释：这个 bug 不会报错，
        # 只会让"泛化误差"悄悄变成拟合误差，然后一路写进 README。
        for dtr, dva in zip(subsets, val_subsets):
            overlap = set(dtr.episode_lengths) & set(dva.episode_lengths)
            assert not overlap, f"{dtr.task} 训练/留出集重叠：{sorted(overlap)[:5]}"
        # shuffle 而不是顺序读：多任务的 val 是三个数据集拼起来的，顺序读会让
        # 前 n 个 batch 全部落在第一个任务里，val loss 只反映那一个任务。
        # 生成器每次 val 前重新播种，所以取到的始终是同一批样本。
        val_gen = torch.Generator()
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=True, generator=val_gen,
            num_workers=min(2, cfg.num_workers), collate_fn=collate, drop_last=True,
            persistent_workers=False, pin_memory=True,
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
    # 上一次是发散退出的，就不要从 last.pt 续跑 —— 那份权重已经坏了，
    # 续上去只会接着发散，而且看起来像"又白跑了一轮"。清掉标记从头开始。
    diverged_mark = out_dir / "DIVERGED.txt"
    if diverged_mark.exists() and cfg.get("resume", True):
        print(f"\n上一轮在此目录发散过（见 {diverged_mark.name}），本次从第 0 步重新开始。",
              flush=True)
        diverged_mark.unlink()
        if last_path.exists():
            last_path.rename(out_dir / "last_diverged.pt")
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
    # 发散看门狗。2026-09-12 有一轮 FM 在第 35400 步开始渐进失稳：梯度范数从 3.75
    # 一路爬到 1.1e6，验证 loss 从 0.158 涨到 0.713，而训练照常跑完 60000 步、
    # 照常存了 checkpoint、照常被评测（三个任务 3%/超时），白花了一小时。
    # 症状在验证 loss 上早就看得见，只是没有人看。
    best_val, bad_val_checks = float("inf"), 0
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

            if val_loader is not None and step % cfg.get("val_every", 2500) == 0:
                vl = val_loss(policy, val_loader, cfg, val_gen,
                              cfg.get("val_batches", 20))
                print(f"  [val] step {step:6d}  loss {vl:.4f}", flush=True)
                if use_wandb:
                    wandb.log({"val/loss": vl, "step": step})

                # 看门狗：验证 loss 连续两次高出历史最低点一半以上就判为发散。
                # 用验证 loss 而不是训练 loss，是因为训练 loss 被 batch 噪声盖着，
                # 涨起来不明显；验证 loss 固定了数据与 RNG，是一条干净的曲线。
                # 两次而不是一次，是为了不被单次抖动误伤；只在训练过半程之前不生效
                # （早期 loss 本来就在剧烈下降，best 还没稳定）。
                ratio = cfg.get("divergence_ratio", 1.5)
                prev_best = best_val          # 判据用的是**更新前**的最低点
                bad_val_checks = diverging(vl, best_val, bad_val_checks,
                                           step, cfg.steps, ratio)
                best_val = min(best_val, vl)
                if bad_val_checks >= 2:
                    msg = (f"发散：验证 loss {vl:.4f} 连续两次高于历史最低 "
                           f"{prev_best:.4f} 的 {ratio} 倍，在第 {step} 步中止。\n"
                           f"梯度范数请看上面的 |g| 一列；这一轮的权重不可用。")
                    print(f"\n{msg}", flush=True)
                    (out_dir / "DIVERGED.txt").write_text(msg + "\n")
                    if use_wandb:
                        wandb.finish(exit_code=2)
                    raise SystemExit(2)

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
