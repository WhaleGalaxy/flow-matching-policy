# 进度与下一步

更新于 2026-09-09。

## 当前状态

主线代码**全部打通并验证过**，第一轮训练暴露的根因已定位、修复并**在 rollout 上确认生效**。
数据就绪，正式训练队列进行中。

| 模块 | 状态 | 备注 |
|---|---|---|
| 环境 | ✅ | conda `robot` (py3.10) + torch 2.5.1+cu121，RTX 4060 8GB 可用 |
| GPU 驱动 | ✅ | 计算栈 + Vulkan ICD，屏蔽 `nvidia_drm` 保住显示栈，见 `docs/gpu-setup.md` |
| 理论笔记 | ✅ | `docs/flow_matching.md` |
| 编码器 | ✅ | DINOv2 / SigLIP / Proprio，冻结 99.5% 参数 |
| FM Denoiser | ✅ | AdaLN + cross-attn + 零初始化，6.45M |
| FMPolicy / DDPMPolicy / BCPolicy | ✅ | 同骨干同参数量（6.80M），对比可控 |
| EMA / CFG / Euler & DDIM 采样 | ✅ | |
| 训练脚本 | ✅ | Hydra + BF16 + 训练前检查 + 周期评测 |
| 评测 / 消融 / 报告脚本 | ✅ | 消融复用 checkpoint，无需重训 |
| 单元测试 | ✅ | 11 个全通过 |
| **数据（pd_ee_delta_pose）** | ✅ | 三任务共 331,283 样本，维度一致 (act=7, proprio=9) |
| 正式训练结果 | 🔄 | 队列进行中：FM / BC / DDPM / 多任务 / seed 123 |

### 第一轮训练的结论

用演示原生的 `pd_joint_pos` 训练，成功率只有 4%。根因是动作表示的信噪比：
该表示下 98.4% 的动作方差仅由"手臂当前在哪"决定，任务信号只有 1.6%。
完整排查过程见 `docs/debugging.md`。

改用 `pd_ee_delta_pose` 后，三个任务的任务信号分别为：

| 任务 | 样本数 | 任务信号 |
|---|---|---|
| PickCube-v1 | 78,465 | 69.1% |
| StackCube-v1 | 108,260 | 74.2% |
| PegInsertionSide-v1 | 144,558 | 57.8% |

相比原来的 1.6% 提升了约 40 倍。三任务的 `act_dim=7` / `proprio_dim=9` 一致，
多任务训练可直接用。

## 修复已验证生效（2026-09-09）

`pd_ee_delta_pose` 的修复确认有效。8000 步的 FM 策略：

| | 第一轮 (pd_joint_pos) | 现在 (pd_ee_delta_pose) |
|---|---|---|
| TCP→方块距离 | 0.163 → 0.22 m（越走越远） | 0.167 → 0.036 m（最好 0.007） |
| 抓到过方块 | 从未 | 70% |
| 方块最大抬升 | — | 125 mm |
| 成功率 | 4% | 10% |

策略已经学会接近、抓取、抬起，只差最后的放置段。成功率低不等于没学到。

### 判据本身要改：成功率在低分区间读不出趋势

原来的判据是"SR 必须随训练步数上升"。**在 `eval.n_episodes=25` 下这个判据是
读不出来的**：8% 就是 2 次成功、4% 就是 1 次，两者的置信区间几乎完全重叠。
实测 4000 步 8.0%、8000 步 4.0%，看上去像在下降，实际只是噪声——差点据此
误判整个方向。

改用连续量，10 个 episode 就能定论：

```bash
python scripts/diagnose_rollout.py --ckpt outputs/fm_PickCube_s42/ckpt_8000.pt
```

**判断标准**：`dmin` 明显小于 `d0` → 策略在朝目标走；`dmin ≈ d0` 或更大 → 没学到。
训练中途的 SR 只当粗略参考，可信的数字来自最终 100-episode 评测。

### 之后的实验队列

按优先级，每轮约 45 分钟：

1. `python -m src.train model=bc` — 行为克隆基线
2. `python -m src.train model=ddpm` — Diffusion Policy 基线（同骨干）
3. `python -m src.train tasks="[PickCube-v1,StackCube-v1,PegInsertionSide-v1]"` — 多任务
4. 多 seed：`++seed=123` / `++seed=456`

消融**不需要重新训练**，复用 checkpoint 即可：

```bash
python scripts/run_eval.py --ckpt <ckpt> --sweep-steps 1 2 5 10 20      # 采样步数
python scripts/run_eval.py --ckpt <ckpt> --sweep-guidance 1.0 1.5 2.0   # CFG 权重
python scripts/benchmark_inference.py                                    # 推理延迟
python scripts/make_report.py                                            # 汇总成表和图
```

只有 proprio 消融（`++use_proprio=false`）需要重训。

## 待定的判断

- **训练步数是否够**：第一轮 20000 步时 loss 已平；但换表示后难度变了，
  可能需要更多步。看 SR 曲线是否还在上升来决定。
- **PegInsertion 可能仍然很难**：它是三个任务里最精密的，而我们只用了 `base_camera`。
  若成功率过低，可考虑单独为它接入 `hand_camera`（该任务的演示里有），
  代价是多任务的观测空间不再统一。
- **训练速度**：目前 7.4 it/s，其中约 88% 花在冻结的 DINOv2 前向上。
  若要跑很多轮消融，可以把特征预先算好缓存（代价是失去像素级数据增强）。

## 已知遗留

- 第一轮（pd_joint_pos）的 checkpoint 已归档到 `outputs/archive_v1_pd_joint_pos/`
  ——它和新一轮的 `run_name` 完全同名（`fm_PickCube_s42`），不挪走会新旧混在一个
  目录里。每个 1.1GB，确认不需要后可整个删掉；后续 checkpoint 已降到 27MB。
- 旧的 `pd_joint_pos` 数据（约 7.7GB）已无用，可删：
  `rm ~/.maniskill/demos/*/motionplanning/trajectory.rgb.pd_joint_pos.physx_cpu.h5`
  （demo 目录当前占 16GB，磁盘剩余 115GB，不删也不影响）
- WandB 尚未接入（`++wandb.enabled=true` 需先 `wandb login`）。
