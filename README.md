# Multi-Task Flow Matching Policy for Robot Manipulation

语言 / Language: **中文** · [English](#english)

用 Flow Matching 训练的语言条件多任务机械臂操作策略，在 ManiSkill3 的三个任务上评测，
并与同骨干的 Diffusion Policy 和行为克隆基线做受控对比。

> 结果表与图待训练完成后填入。

## 核心思路

一句话：**把"预测下一段动作"建模成从噪声到动作块的一次流动，而不是一次回归。**

- **为什么不用回归（BC）**：演示数据里同一个观测常常对应多种合法动作。MSE 回归的
  最优解是这些动作的**条件均值**，而均值往往是非法动作。
  [实测](notebooks/w1_overfit_and_multimodal.py)：双模态目标下 FM 覆盖两个模态 100%，
  BC 覆盖 0%（输出恒为两模态的中点）。
- **为什么不用扩散（DDPM）**：Flow Matching 回归的条件速度场是**常数** `x₁ − x₀`，
  条件路径为直线，因此少步 Euler 积分即可采样，而 DDPM 通常需要 100 步。
  本仓库实现了**同骨干、同参数量**的 DDPM 基线，把延迟差异归因到建模选择本身。

理论推导（含 CFM 梯度等价性证明）见 [docs/flow_matching.md](docs/flow_matching.md)。

## 架构

```
RGB (2帧, 128×128)  ──► DINOv2-S (冻结) ──► 81 个 patch token ─┐
语言指令             ──► SigLIP 文本塔(冻结) ──► 1 个 token ────┼─► context (B, 83, 256)
关节角 + 夹爪 (9维)  ──► MLP ─────────────► 1 个 token ────┘         │
                                                                      │ cross-attn
噪声动作块 x_t (16×7) ──► Transformer denoiser (AdaLN 注入流时间 t) ◄──┘
                                    │
                                    ▼
                         速度场 v_θ ──► Euler 积分 (10步) ──► 动作块
```

- 视觉/语言骨干全部冻结（**99.5% 参数冻结**）：演示只有 1000 条/任务，微调 ViT 必然过拟合
- denoiser 用 DiT 式设计：AdaLN 注入流时间，残差分支与输出层**零初始化**，起步即恒等映射
- checkpoint 同时保存在线权重与 EMA 权重，用哪套由实测决定而非默认（见下方工程记录）
- CFG 的有条件/无条件两路拼成一个 batch，一次前向完成

## Quick Start

```bash
conda create -n robot python=3.10 -y && conda activate robot
pip install -r requirements.txt

# 混合显卡笔记本上装驱动会让 Xorg 选错显卡导致开机黑屏，见 docs/gpu-setup.md
sudo bash scripts/setup_gpu.sh

# 1) 下载并重放演示数据（官方 demo 的 obs/ 是空的，必须重放渲染才有图像）
bash scripts/download_demos.sh
bash scripts/replay_demos.sh

# 2) 训练
python -m src.train                                    # FM，单任务 PickCube
python -m src.train model=bc                           # 行为克隆基线
python -m src.train model=ddpm                         # Diffusion Policy 基线
python -m src.train tasks="[PickCube-v1,StackCube-v1,PegInsertionSide-v1]"   # 多任务

# 3) 评测与消融（消融复用同一个 checkpoint，无需重训）
python scripts/run_eval.py --ckpt outputs/fm_PickCube_s42/ckpt_20000.pt --n-episodes 100
python scripts/run_eval.py --ckpt ... --n-episodes 100 --no-ema      # 在线权重对照
python scripts/run_eval.py --ckpt ... --sweep-steps 1 2 5 10 20
python scripts/run_eval.py --ckpt ... --sweep-guidance 1.0 1.5 2.0
python scripts/make_report.py --weights ema                          # 或 online / both

# 成功率在低分区间没有分辨率时，用连续指标判断策略是否在学
python scripts/diagnose_rollout.py --ckpt outputs/fm_PickCube_s42/ckpt_20000.pt

# 4) 推理延迟基准
python scripts/benchmark_inference.py
```

## 结果

<!-- RESULTS_TABLE -->

## 工程记录

开发过程中发现的、与常见教程/计划不符的地方，都在代码和笔记里留了记录：

| 发现 | 说明 |
|---|---|
| DINOv2 输入不能是 96×96 | patch 是 14×14，96 除不尽。改用 126×126 → 81 个 patch |
| 动作表示决定成败 | `pd_joint_pos` 下 98.4% 的动作方差只由"手臂当前在哪"决定，任务信号仅 1.6%，成功率锁死在 4%。换成 `pd_ee_delta_pose` 后任务信号升至 57–74%（约 40 倍），策略才学会接近与抓取。详见 [docs/debugging.md](docs/debugging.md) |
| 动作维度随控制模式变 | `pd_joint_pos` 是 8 维（Panda 7 关节 + 夹爪），`pd_ee_delta_pose` 是 7 维（6 自由度增量 + 夹爪） |
| 成功率在低分区间读不出趋势 | 25 个 episode 下 8% 和 4% 只差一次成功，置信区间几乎重合。用 TCP→目标距离、抓取率这类连续量判断，10 个 episode 即可定论（`scripts/diagnose_rollout.py`） |
| EMA 未必更好 | `decay=0.9999` 的时间常数是 10000 步，对 20000 步的训练太慢，实测在线权重的抓取率 80% vs EMA 45%。两套权重都存进 checkpoint，评测时再选 |
| 官方 demo 的 `obs/` 是空的 | 采集时 `obs_mode="none"`，必须重放渲染才有图像观测 |
| BF16 不需要 GradScaler | GradScaler 是给 FP16 防梯度下溢的；BF16 指数位与 FP32 相同，不会下溢 |
| FM 的 loss 不会趋近 0 | 下界是条件方差 `E[Var(x₁−x₀｜x_t,t,ctx)]`，不能拿"loss < 0.1"当验收标准 |
| TF32 默认是关的 | torch 2.x 里 matmul 的 TF32 默认关闭，打开实测有 1.3–2.9x 加速 |
| checkpoint 曾达 1.1GB | 冻结骨干被存了两份；只存可训练参数后降到 27MB |

## 已知局限

- 只用 `base_camera`。StackCube/PegInsertion 的演示里还有腕部相机，但 PickCube 没有，
  多任务需要统一观测空间。接入腕部相机应能明显改善 PegInsertion。
- 使用独立耦合而非 minibatch OT 耦合。完整的 OT-CFM 会在 batch 内解一次最优传输匹配，
  能进一步降低回归目标方差、拉直边缘轨迹。
- 冻结的视觉编码器占了约 88% 的训练时间。因为它是冻结的，特征可以整个数据集预先算好缓存，
  代价是失去像素级数据增强。

## 目录结构

```
fm_policy/
├── configs/          # Hydra 配置：train.yaml + model/{bc,fm,ddpm} + task/{pick,stack,peg}
├── src/
│   ├── models/
│   │   ├── encoders.py   # VisualEncoder / LanguageEncoder / ProprioEncoder
│   │   ├── denoiser.py   # SinusoidalPosEmb / AdaLN / CrossAttention / FMDenoiser
│   │   └── policy.py     # FMPolicy / DDPMPolicy / BCPolicy / EMA / ActionNormalizer
│   ├── data/dataset.py   # ManiskillDataset + 训练评测共用的预处理
│   ├── train.py          # Hydra 训练入口
│   └── evaluate.py       # rollout 与成功率统计
├── scripts/          # 环境搭建、数据重放、评测、报告、延迟基准
├── notebooks/        # 逐步验证实验（玩具 FM、编码器选型、多模态论证）
├── docs/             # 推导笔记 / 调试记录 / GPU 配置排查
└── tests/            # pytest 单元测试
```

---

## English

A language-conditioned multi-task manipulation policy trained with Flow Matching,
evaluated on three ManiSkill3 tasks against same-backbone Diffusion Policy and
behaviour-cloning baselines.

Key idea: model the next action chunk as a *flow* from noise rather than a regression.
BC's MSE optimum is the conditional mean of the demonstrated actions, which is often
itself an invalid action; Flow Matching's constant conditional velocity `x₁ − x₀` gives
straight conditional paths, so few-step Euler integration suffices where DDPM needs ~100.

The DDPM baseline shares the identical encoder and denoiser (same parameter count), so
the latency and quality differences are attributable to the modelling choice alone.

See [docs/flow_matching.md](docs/flow_matching.md) for the derivation, including the
proof that the conditional and marginal Flow Matching objectives have identical gradients.
