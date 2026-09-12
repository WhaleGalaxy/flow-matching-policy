# 概念问答记录

这份文件记录我在做这个项目时问过的概念性问题和简要答案，按提问顺序累加。
和 [debugging.md](debugging.md)（排查过程）、[status.md](status.md)（进度）分工不同：
这里只放"这是什么、为什么这么做"，不放实验数据。

---

## 1. 当前网络的结构是什么样的？

一句话：**冻结的视觉/语言骨干把观测编码成 token 序列，一个小 Transformer 以这些
token 为条件、把噪声动作块去噪成 16 步的动作序列。**

```
RGB (2 帧, 126×126)   ──► DINOv2-S(冻结) ──► 81 个 patch token ─┐
语言指令               ──► SigLIP 文本塔(冻结) ──► 1 个 token ───┼─► context (B, 83, 256)
本体状态 (9 或 12 维)  ──► 3 层 MLP ──────────► 1 个 token ───┘        │
                                                                       │ cross-attn
噪声动作块 x_t (16×7) ──► Transformer denoiser (AdaLN 注入流时间 t) ◄──┘
                                     │
                                     ▼
                          速度场 v_θ ──► Euler 积分 ──► 动作块 (16×7)
```

**三个编码器**（[src/models/encoders.py](../src/models/encoders.py)）统一输出 256 维
token，拼成 context：

| 编码器 | 输入 | 输出 | 是否训练 |
|---|---|---|---|
| `VisualEncoder` | 2 帧 126×126 RGB | 81 个 patch token（126/14=9，9×9） | 冻结，只训最后的线性投影 |
| `LanguageEncoder` | 任务指令字符串 | 1 个 token（带缓存） | 冻结，只训投影 |
| `ProprioEncoder` | 关节角+夹爪（+goal_pos） | 1 个 token | 训练 |

两帧在时间维上**取平均**而不是拼接，所以 token 数不随观测窗口长度增长；
用 patch token 而不是 CLS token，是因为操作任务要知道物体在哪，池化会丢掉空间信息。

**denoiser**（[src/models/denoiser.py](../src/models/denoiser.py)）是 DiT 式的 4 层
Transformer，`d_model=256`、`n_heads=8`。每层依次做：自注意力（动作块内部的时序关系）
→ 交叉注意力（去看 context）→ FFN，三个分支都由 AdaLN 按流时间 t 调制。
残差分支和输出层零初始化，所以训练起步时整个网络是恒等映射。

**三种策略头**（[src/models/policy.py](../src/models/policy.py)）共用上面这套编码器：

- `FMPolicy`：denoiser 回归速度场 `x₁ − x₀`，推理时 Euler 积分 10 步
- `DDPMPolicy`：**同一个 denoiser 结构、同样的参数量**，只是回归噪声 ε，推理 DDIM 100 步
- `BCPolicy`：context 平均池化后过 3 层 MLP 直接回归动作块。默认 `hidden=1024`
  只有 1.77M 参数，要做等容量对照必须显式给 `++model.hidden=2364`

参数量：FM 与 DDPM 各 6.80M 可训练参数，BC 6.81M；冻结部分占总参数的 99.5%。
冻结的原因是每个任务只有 1000 条演示，微调 ViT 必然过拟合。

关键的形状约束：DINOv2 的 patch 是 14×14，所以输入分辨率必须被 14 整除
（96 不行，用 126）；动作维度 7 = 3 位移 + 3 旋转 + 1 夹爪，由控制模式
`pd_ee_delta_pose` 决定，换控制模式这个数就变。

---

## 2. 使用的 W&B 有什么用？

Weights & Biases 是个训练过程的记录网站。训练脚本每 100 步把 loss、学习率、
梯度范数发上去，每次评测把成功率发上去，在网页上看曲线。

在本项目里它做三件事：

1. **跨 run 对比曲线**。FM / DDPM / BC 三条训练曲线叠在一张图上，不用自己存 log 画图。
2. **记录每次 run 的完整配置**。`wandb.init(config=...)` 把整份 Hydra 配置存下来，
   两个月后看到某条曲线还能知道当时 `use_goal` 是什么、跑了多少步。
3. **远程看训练**。跑着 45 分钟的训练时不用守着终端。

**本机已配置好，默认开启。** API key 存在 `~/.netrc`（由 `wandb login` 写入，
在仓库外，不会被提交），`configs/train.yaml` 里 `wandb.enabled: true`。
直接 `python -m src.train` 就会上传，run 出现在
`https://wandb.ai/models-national-university-of-singapore/fm-policy`。

不想上传某一次训练时用 `++wandb.enabled=false` 关掉；
想离线跑、之后再同步用 `WANDB_MODE=offline`，回头 `wandb sync wandb/offline-run-*`。

每个 run 自动带上模型名和任务名作为标签，网页上可以按这两个筛。
默认 entity 是团队 `models-national-university-of-singapore`；
想发到个人账号名下就把 `wandb.entity` 写成用户名。

一个重要的提醒：**W&B 上的 loss 曲线漂亮不代表策略能用**。第一轮训练 loss 从 2.06
收敛到 0.067，成功率却只有 4%，根因是动作表示的信噪比只有 1.6%
（见 [debugging.md](debugging.md)）。而且 FM 与 DDPM 的 loss 目标不同、量纲不可比，
两条曲线叠在一起看没有意义。曲线只能看训练有没有崩，验收要看 `eval/*` 里的成功率。

---

## 3. 常说的 pipeline 是什么意思？

字面是"流水线"，指**一串固定顺序的处理步骤，前一步的输出就是后一步的输入**。
说"跑一遍 pipeline"就是说"从头到尾把这串步骤走一遍"，而不是只跑其中某一个脚本。

这个词在不同层面上都用，本项目里至少有三层：

**实验 pipeline**（整个项目的主流程，见 README 的 Quick Start）：

```
下载演示 → 重放渲染出图像 → 训练前检查 → 训练 → 评测/消融扫描 → 分析出图 → 汇总报告
```

**数据 pipeline**（一条样本从磁盘到模型）：

```
HDF5 轨迹 → 切成 (2 帧观测, 16 步动作) 的窗口 → 图像 resize/归一化 → 动作归一化 → batch
```

**推理 pipeline**（机器人跑一步）：

```
相机图像 + 本体状态 → 编码成 context → 采样出 16 步动作块 → 执行前 8 步 → 重新观测
```

用这个词通常是想强调两件事：**步骤之间有依赖**（跳过重放渲染，训练时读到的图像是空的），
以及**它是可以被整体自动化和整体复现的**。

---

## 4. 如果我要手动跑训练的话应该在终端里输入什么？

先激活环境，进项目目录：

```bash
conda activate robot
cd ~/flow_matching/fm_policy
```

**默认配置（Flow Matching，PickCube，30000 步）：**

```bash
python -m src.train
```

**受控对比用的三条命令**（README 里的标准配置，20000 步 + 接入 goal_pos）：

```bash
python -m src.train ++use_goal=true ++steps=20000
python -m src.train model=ddpm ++use_goal=true ++steps=20000
python -m src.train model=bc ++model.hidden=2364 ++use_goal=true ++steps=20000
```

命令行语法是 Hydra 的：

- `model=ddpm` 是**换一整个配置组**，对应 `configs/model/ddpm.yaml`
- `++key=value` 是**覆盖单个字段**，`++` 表示这个 key 原本可能不存在也允许加
- 想改什么直接查 `configs/train.yaml` 里的字段名，例如
  `++batch_size=64`（显存不够时）、`++seed=123`、`++tasks=[StackCube-v1]`、
  `++wandb.enabled=false`（这一次不上传 W&B）

**跑之前要确认的三件事**：演示数据已经重放过（`bash scripts/download_demos.sh`
和 `bash scripts/replay_demos.sh`，官方 demo 的 `obs/` 是空的），GPU 可用，
以及自己看一眼训练开头自动打印的"训练前检查"里的动作表示信噪比。

**输出去哪**：`outputs/<model>_<task>_s<seed>/`，例如 `outputs/fm_PickCube_s42/`，
里面每 5000 步存一个 `ckpt_*.pt`（在线权重 + EMA 权重都存）。

**长时间训练建议挂后台**，否则关掉终端就断了：

```bash
nohup python -m src.train ++use_goal=true ++steps=20000 > train_fm.log 2>&1 &
tail -f train_fm.log
```

训练完评测（注意 `--no-ema` 不是笔误，本项目 EMA 的时间常数对 20000 步太慢）：

```bash
python scripts/run_eval.py --ckpt outputs/fm_PickCube_s42/ckpt_20000.pt --n-episodes 100 --no-ema
```
