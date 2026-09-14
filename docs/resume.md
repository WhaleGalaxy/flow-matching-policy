# 简历素材

数字口径：ManiSkill3 / SAPIEN，运动规划演示，单个 128×128 `base_camera`，
100 episode 闭环评测，在线权重，`guidance=1.0`，`execute_horizon=8`，
`max_episode_steps=300`。训练集与评测用的留出集按 episode 划分（`val_frac=0.1`,
`split_seed=0`）。复核入口：`outputs/2026-09-12-heldout/multitask.csv`（多任务）、
`pick.csv`（单任务）、`reference_100steps.csv`（对齐官方 100 步预算的重测）。

除标注 *(n=1)* 的格子外，全部为 3 个随机种子的 mean±std。

---

## 写法的原则

这个项目的产出是**一次三方对比**，以及从对比里得出的**各自适用范围与选择依据**。
所以顺序是：

> 造了什么 → 实现了哪三种策略 → 对比出了什么 → 怎么判断该选哪个 → 工程与严谨性

不要把某一种方法写成"我的方法"。三种策略各有自己赢的场合，能说清"什么时候用
哪个、为什么"，比宣称某一个更强更难做到，也更经得起追问。

---

## 项目标题

> **语言条件的多任务机械臂操作策略：三类动作头的等条件对比**
> （ManiSkill3 / SAPIEN）

---

## 要点条目

**1. 从零搭建完整的模仿学习管线**

> 演示数据重放渲染、动作分块数据集、动作归一化、bf16 训练循环、EMA、断点续跑、
> Hydra 配置、闭环 rollout 评测。冻结 DINOv2 提取 patch token、SigLIP 文本塔
> 编码指令，DiT 式 Transformer 动作头以 cross-attention 读取条件、AdaLN 注入
> 流时间，一次生成 16 步动作块，在三个操作任务上闭环控制。

**2. 从零实现三类动作头，并在三个任务上做等条件对比**

> Flow Matching（Euler 积分采样）、Diffusion Policy（DDPM 训练 + DDIM 采样 +
> classifier-free guidance）、Transformer 回归基线（cross-attention 读出）。
> 三者共用同一冻结编码器、同一参数量（6.8M）、同一训练预算（60k 步）、
> 同一观测配置、同一留出划分，评测用完全相同的初始场景。

**3. 对比结论：三种策略的适用范围各不相同**

这是项目的主产出。三张表说的不是同一件事。

> **单任务（PickCube，3 seed）**
>
> | 动作头 | 成功率 |
> |---|---|
> | Flow Matching | **61 ± 15%** |
> | 回归基线（cross-attn） | 53 ± 19% |
> | Diffusion Policy | 13 ± 9% |
> | 回归基线（mean-pool） | 3% *(n=1)* |
>
> **多任务（三任务同时训练，3 seed）**
>
> | 动作头 | PickCube | PushCube | StackCube |
> |---|---|---|---|
> | 回归基线（cross-attn） | **47 ± 28%** | 92 ± 2% | **41 ± 8%** |
> | Flow Matching | 33 ± 28% | **95 ± 5%** | 18 ± 11% |
> | Diffusion Policy | 6 ± 3% | 95 ± 2% | 1 ± 1% |
>
> 三条可直接用于选型的结论：
>
> - **单任务、要精度 → Flow Matching。** 它在 PickCube 上领先回归基线 8 个点，
>   领先 Diffusion 48 个点。
> - **多任务 → 等条件的回归基线不输，且更便宜。** 同骨干同预算下它在两个较难的
>   任务上反超，而推理只要一次前向。
> - **视觉上已经把目标给足的任务（PushCube）→ 选谁都一样。** 三者都在 92–96%，
>   这一格不该作为任何方法的证据。
> - **Diffusion 的弱是可复现的，不是抽到坏种子。** 三个种子在 PickCube 上是
>   8 / 7 / 3%（±3），而同样三个种子下 FM 是 13 / 22 / 65（±28）、BC 是
>   28 / 35 / 79（±28）。"你的基线是不是没调好"这个质疑，答案是三个种子跑出来都一样。

**4. 推理成本相差一个数量级，且这是可达控制频率的硬约束**

> 在同一 checkpoint 上扫采样步数（单任务 PickCube，全量数据）：Flow Matching
> 1 / 2 / 4 / 8 / 16 步是 67 / 74 / 72 / 69 / 75%，**两步即达到它自己的最优**；
> 同条件的 Diffusion Policy 是 2 / 2 / 3 / 10 / 25 / 33 / 33%（1→100 步），
> 要 50 步才到它自己的最优 33%。**同等成绩下采样步数差一个数量级**，
> 而采样步数直接换算成可达控制频率。

**5. 从对比里沉淀出「训练前就能判断该选什么」的方法**

这一条是最难被外行找出来的部分，面试时也最经得起追问 —— 每一条判据都只要
数分钟，而每一条各可否决一整轮 40 分钟的训练。

> **动作表示的信噪比。** `pd_joint_pos` 下 98.4% 的动作方差只反映"手臂当前在
> 哪"，任务信号仅占 1.6%，成功率锁死在 4%。改用末端增量控制后任务信号升至
> 57–74%，策略才学会接近与抓取。
>
> **观测充分性探针。** 抓取率 95% 但成功率 6%。线性探针显示目标位置在相机中的
> 可恢复性 R²≈0（每局随机的不可见标记），接入观测后成功率提升一个数量级。
> "高抓取率 + 低成功率"是观测缺失的典型指纹。
>
> **条件动作分布的模态检验 —— 决定要不要上生成式动作头。** 在真值状态空间取
> 跨轨迹最近邻、投影到主方向，双峰系数 0.49 / 0.39 / 0.48，而正态参照是 0.555。
> 这批演示的条件动作分布是单峰的：**选生成式动作头的前提在这份数据上不成立**，
> 这解释了为什么多任务上回归基线不输。

**6. 定位并修复三处使对比本身失效的缺陷**

> 回归基线把 83 个 context token 平均池化，目标被稀释为 1/83；改为与生成式动作头
> 逐层同构的 cross-attention 读出后，基线从 3% 升到 52% —— 原先报告的 71 个
> 百分点领先里，49 个来自读出方式而非建模方式。扩散基线曾在所有配置下锁死于
> 2.0%，根因是 cosine 调度使 `ᾱ` 末项降到 2.4e-7、DDIM 恰好从该项起步，
> 把噪声预测误差放大约两千倍。汇总脚本未把训练配置纳入分组键，把 11 个不同
> 配置的实验平均成了一个不对应任何实验的数字。

**7. 严谨性：把"这个差距能不能信"当成一等问题**

> **先量噪声再下结论。** PickCube 的种子间标准差实测 ±28 个百分点（同一配置三个
> 种子跑出 13 / 22 / 65%），据此把全部结论改为 3 seed 报告，并对区分不出的格子
> 明确写"分不出来" —— 上表里 Flow Matching 与回归基线在 PickCube 上就是分不出来。
> 同配置重复评测的跨度（4% vs 8%）作为测量噪声下限单独记录。
>
> **留出验证集 + 发散看门狗。** 按 episode 划分 10% 留出集，训练中每 2500 步测
> 验证 loss；连续两次高于历史最低的 1.5 倍即中止并打标记，该轮结果自动排除出
> 结果表。加宽的回归基线正是靠这条被直接观测到过拟合：验证 loss 在第 10000 步
> 见底后回升 4.6%，是四条臂里唯一掉头的。
>
> **主动证伪自己的解释。** 曾把"加宽后成功率下降"解释为生成式动作头特有的代价，
> 随后补上等宽度的回归基线对照：它掉得更多（79 → 36，对 Flow Matching 的
> 65 → 48），原解释不成立，已在项目里改写。

---

## 工程与基础设施（可并入某一条，或用于回答「工程能力」类提问）

> 冻结视觉骨干占约 88% 训练时间且管线无图像增强，据此预计算并缓存 DINOv2 patch
> token（与像素路径相对误差 2.2e-4，低于训练所用 bf16 精度），训练吞吐由
> 8.3 提升至 25–36 it/s，使 3 seed × 多配置的对比矩阵在单卡上可行；训练支持含
> 优化器状态的断点续跑，实验以阶段级 `.done` 标记的串行队列调度，意外中断最多
> 损失 1000 步；内存守护进程在可用内存连续跌破阈值时停队列而不是让内核杀掉
> 用户会话。

---

## English

**Summary (≈60 words)**

> Language-conditioned multi-task manipulation on ManiSkill3 (frozen DINOv2 + SigLIP,
> DiT-style action head, 6.8M trainable parameters). Implemented three action heads —
> Flow Matching, Diffusion Policy, and a regression baseline — and compared them under
> a matched encoder, parameter count, training budget, and held-out split. Established
> where each one wins, and distilled the comparison into minutes-long pre-training
> checks that predict the right choice before training.

**Bullets**

> - Built a full imitation-learning pipeline from scratch (demo replay and rendering,
>   action chunking, bf16 training loop with EMA and resumable checkpoints, Hydra
>   configs, closed-loop evaluation) and implemented three action heads on a shared
>   frozen backbone: Flow Matching, Diffusion Policy (DDPM training, DDIM sampling,
>   classifier-free guidance), and a cross-attention regression baseline.
> - Ran a matched-condition three-way comparison over three tasks with a held-out split
>   and three seeds, and established that the three heads win in different regimes:
>   Flow Matching leads single-task PickCube (61±15% vs 53±19% regression and 13±9%
>   diffusion); the regression baseline matches or beats it multi-task (47±28/92±2/41±8%
>   vs 33±28/95±5/18±11%) at one forward pass per action chunk; and on a task whose
>   goal is already visible all three land at 92–96%, so that cell is evidence for
>   nothing. The Diffusion arm's weakness is reproducible rather than an unlucky seed:
>   8 / 7 / 3% on PickCube across the same three seeds where Flow Matching gives
>   13 / 22 / 65% and the regression baseline 28 / 35 / 79%.
> - Quantified the sampling-cost gap on a single checkpoint: Flow Matching is flat from
>   1 to 16 Euler steps (67 / 74 / 72 / 69 / 75%), reaching its own optimum at two
>   steps, while the matched Diffusion Policy goes 2 / 2 / 3 / 10 / 25 / 33 / 33% from
>   1 to 100 DDIM steps, needing 50 to reach its own optimum of 33% — an order of
>   magnitude in sampling cost, which sets the achievable control frequency.
> - Turned the comparison into three pre-training criteria, each minutes to run and each
>   able to veto a 40-minute training run: action-representation SNR (task signal 1.6%
>   → 57–74% after switching to end-effector deltas, success 4% → order of magnitude
>   higher), an observation-sufficiency probe (goal recoverability R²≈0 explains 95%
>   grasp rate against 6% success), and a **conditional-action-distribution modality
>   check** — bimodality coefficients 0.49/0.39/0.48 against a normal reference of
>   0.555, i.e. the premise for a generative action head does not hold on this data,
>   which is why the regression baseline is competitive multi-task.
> - Measured the noise before trusting any gap: seed-to-seed standard deviation on
>   PickCube is ±28 points, so all conclusions are reported over three seeds and cells
>   that cannot be separated are labelled as such. Added a held-out split with a
>   divergence watchdog (validation loss 1.5× above its best, twice in a row, aborts the
>   run and excludes it from the results table), which caught the widened regression
>   baseline overfitting directly: validation loss bottoms at step 10k and rises 4.6%.
> - Found and fixed three flaws that invalidated the comparison itself: a mean-pooling
>   readout that diluted the goal token to 1/83 (baseline 3% → 52%, accounting for 49 of
>   71 points previously claimed), a dead diffusion baseline pinned at exactly 2.0%
>   across all configurations from a cosine schedule leaving ᾱ at 2.4e-7 where DDIM
>   starts, and an aggregation script that averaged 11 differently-configured runs into
>   a number corresponding to no experiment.

---

## 面试问答准备

**「三种策略你会怎么选？」**

> 先看任务：目标在相机里已经给足的任务，三者都是 92–96%，选最便宜的回归基线。
> 单任务要精度，Flow Matching 领先 8 个点。多任务，等条件的回归基线不输而且
> 推理只要一次前向。Diffusion 在我这套设置下三个任务都最差，而且最吃数据。
>
> 但真正的答案是**训练前就能判**：去量演示的条件动作分布是不是多峰的。
> 我这批运动规划演示双峰系数 0.49/0.39/0.48、正态参照 0.555，是单峰的，
> 所以生成式动作头的前提不成立 —— 这正是多任务上回归基线不输的原因。
> 这个检验几分钟就能跑完，省掉的是一轮 40 分钟的训练。

**「多任务上回归基线反超，你怎么解释？」**

> 依次排除：CFG dropout 的训练条件不对称（补了 0.2 的对照臂，8/96/1 对 7/98/0，
> 没有影响）、采样步数不足（2 到 20 步曲线是平的）、采样过度发散（模型散度
> 0.325 对数据 0.42，校准良好）、推理时抽样本而非取均值（取 16 个样本平均反而
> 更差）。然后做开环测量，发现流匹配估计条件均值的误差是回归的 1.5 倍。
> 最后去量数据，条件动作分布是单峰的 —— 单峰数据上你要的是均值，
> 而回归基线直接就在估均值。

**「你怎么知道这些差距不是噪声？」**

> 因为我先量了噪声。PickCube 同一配置三个种子是 13 / 22 / 65%，标准差 ±28 个
> 百分点。所以我不说 Flow Matching 在 PickCube 多任务上输给回归基线 —— 33±28
> 对 47±28 是分不出来的，我在表里就这么写。能说的是单任务那一格（61±15 对
> 53±19）和 Diffusion 那一行（13±9，差了 48 个点）。
>
> 同配置重复评测本身的跨度我也记了：同一个配置跑两次是 4% 和 8%，
> 这是这套评测的测量噪声下限。

**「这个项目最难的部分？」**

> 不是训练，是让对比成立。我最初报过 71 个百分点的领先，其中 49 个后来被证明
> 来自基线的读出方式而不是建模方式。类似的事不止一次：汇总脚本把 11 个不同配置
> 的实验平均成一个数、队列里 8 个手工建的完成标记让两组训练被静默跳过、
> Hydra 的重复参数取最后一个导致一条对照臂实际跑的是对照组本身。
> 现在这些都有记录和防护 —— 结论对不对，取决于有没有人去查它。

**「哪里没做成？」**

> 三条。多任务的 Diffusion 臂目前只有一个种子，不能和另外两条并排比，正在补。
> 想构造真正多模态的演示来验证「生成式动作头在多模态数据上更强」这条推论，
> 混合运动规划与 PPO 两种解法后条件散度到 1.05，但双峰系数只有 0.547，
> 仍是宽的平顶单峰，所以这条推论**没有**被证实，只证实了它的前提在单峰数据上
> 不成立。评测预算上，本项目用 300 步而 ManiSkill 官方基线脚本用 100 步，
> 已用 100 步重测一遍以便与已发表数字对齐，两份都保留。
