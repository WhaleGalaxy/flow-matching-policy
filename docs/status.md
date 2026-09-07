# 进度与下一步

更新于 2026-09-08。

## 当前状态

主线代码**全部打通并验证过**，第一轮训练暴露的根因已定位并修复，正在重新准备数据。

| 模块 | 状态 | 备注 |
|---|---|---|
| 环境 | ✅ | conda `robot` (py3.10) + torch 2.5.1+cu121，RTX 4060 8GB 可用 |
| GPU 驱动 | ✅ | headless 计算栈 + Vulkan ICD，显示栈未受影响 |
| 理论笔记 | ✅ | `docs/flow_matching.md` |
| 编码器 | ✅ | DINOv2 / SigLIP / Proprio，冻结 99.5% 参数 |
| FM Denoiser | ✅ | AdaLN + cross-attn + 零初始化，6.45M |
| FMPolicy / DDPMPolicy / BCPolicy | ✅ | 同骨干同参数量（6.80M），对比可控 |
| EMA / CFG / Euler & DDIM 采样 | ✅ | |
| 训练脚本 | ✅ | Hydra + BF16 + 训练前检查 + 周期评测 |
| 评测 / 消融 / 报告脚本 | ✅ | 消融复用 checkpoint，无需重训 |
| 单元测试 | ✅ | 9 个全通过 |
| **数据（pd_ee_delta_pose）** | ✅ | 三任务共 331,283 样本，维度一致 (act=7, proprio=9) |
| 正式训练结果 | ⬜ | **下一步** |

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

## 明天怎么开始

```bash
conda activate robot
cd ~/flow_matching/fm_policy

# 0) GPU 模块每次重启后需重新加载
sudo modprobe nvidia nvidia_uvm && nvidia-smi

# 1) 数据已就绪（三任务已转换并校验，无需重跑）

# 2) 先跑单任务 FM 验证修复（约 45 分钟，20000 步）
#    关键看训练开始时打印的"任务信号"应 ≈70%，以及 SR 是否随训练上升
python -m src.train ++steps=20000 ++eval.every=4000 2>&1 | tee outputs/fm_pick_v2.log
```

**判断标准**：成功率必须**随训练步数上升**。第一轮失败的最强信号就是 SR 在
4k/8k/12k/16k/20k 五个点上精确相等。只要看到 SR 在动，方向就是对的。

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

- `outputs/fm_PickCube_s42/` 下是第一轮（pd_joint_pos）的 checkpoint，
  每个 1.1GB 且已无用，可以删掉；后续 checkpoint 已降到 27MB。
- 旧的 `pd_joint_pos` 数据（约 7.7GB）已无用，可删：
  `rm ~/.maniskill/demos/*/motionplanning/trajectory.rgb.pd_joint_pos.physx_cpu.h5`
  （demo 目录当前占 16GB，磁盘剩余 115GB，不删也不影响）
- WandB 尚未接入（`++wandb.enabled=true` 需先 `wandb login`）。
