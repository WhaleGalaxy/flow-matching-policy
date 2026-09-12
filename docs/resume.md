# 简历素材

所有数字可在 `outputs/2026-09-11/*.csv` 与 `docs/status.md` 复核，
口径统一为 100 episode、在线权重、`guidance=1.0`、`execute_horizon=8`。

---

## 项目标题

**首选**

> **少步流匹配机器人操作策略**（ManiSkill3，语言条件多任务）
>
> Few-Step Flow Matching Manipulation Policy

"少步"三个字把差异化前置了 —— 这是流匹配相对扩散**唯一站得住且实测出来**的
优势，也是机器人岗位真正在意的（推理延迟直接决定控制频率）。
读者扫一眼就知道这不是又一个"我训了个策略"。

**备选**

| 标题 | 适合的场景 |
|---|---|
| 生成式动作头的受控对比与适用判据 | 投研究岗、强调方法学时 |
| 语言条件多任务操作策略：Flow Matching vs Diffusion vs 回归 | 对方看重对比的完整性时 |
| 少步流匹配操作策略与训练前诊断工具链 | 想同时强调工程沉淀时 |

不建议用带 "Multi-Task" 的旧标题：三任务在这个领域不算规模，撑不起这个词。

---

## 项目概述（133 字，简历正文主用）

> 语言条件的多任务机械臂操作策略。Flow Matching 动作头两步采样即达 74%，
> Diffusion 需 50 步才到 33%；三任务 60/96/36% 对 44/92/3%。
> 提出「条件动作分布模态检验」，训练前数分钟即可判定生成式动作头是否值得用，
> 并据此划出本方法的适用边界。

**为什么这样写。** 第一句给能力，第二句给一个**读者能在一秒内感受到**的对比 ——
"两步 74% 对五十步 33%"比"60/96/36 对 44/92/3"冲击力大得多，因为它同时说了
更好和更快。第三句给一个有名字的方法论贡献。最后半句用"划出适用边界"
承接住不及回归基线那件事：**同样诚实，但朝前看，而不是以认输收尾。**

---

## 备选概述

**偏严谨（133 字）** —— 投研究岗、或对方明显重视方法学时

> 从零搭建 Flow Matching / Diffusion / 回归的三方受控对比（同编码器、同参数量、
> 同预算），排出三处使结论失效的缺陷。Flow Matching 以 1/10 采样步数取得三任务
> 60/96/36%，胜 Diffusion 的 44/92/3%；并沉淀出一条训练前判据。

**偏能力（122 字）** —— 投工程岗、对方关心"你能把东西做出来吗"

> 语言条件的多任务机械臂操作策略：Flow Matching 动作头以 1/10 的采样步数超过
> 同骨干 Diffusion Policy（三任务 60/96/36% 对 44/92/3%）。提出「条件动作分布
> 模态检验」，训练前数分钟即可判定生成式动作头是否值得用。

---

## 一句话版（70 字，用于个人简介或项目列表）

> 语言条件多任务 Flow Matching 操作策略，三任务成功率 60/96/36%，
> 优于同骨干 Diffusion Policy 的 44/92/3%。

---

## 要点条目（简历项目经历用，按重要性排序）

选前两条即可撑起一个条目；四条全用适合放在项目主页或作品集。

**1. 方法与主结果**

> 实现语言条件的多任务操作策略：冻结 DINOv2 patch token 与 SigLIP 文本塔编码
> 观测与指令，DiT 式 denoiser 以 cross-attention 读取条件、AdaLN 注入流时间，
> 生成 16 步动作块。同骨干、同 6.8M 参数、同 60k 步训练预算下，三任务
> （PickCube / PushCube / StackCube）成功率 60/96/36%，Diffusion Policy 为
> 44/92/3%；单任务 PickCube 上 Flow Matching 两步采样即达 74%，
> 而 Diffusion 需 50 步才到 33%，**推理成本相差一个数量级**。

**2. 发现并修复三处使结论无效的对照缺陷**

> — 基线读出方式：行为克隆基线将 83 个 context token 平均池化，使目标信息被
>   稀释为 1/83；改为与主方法逐层同构的 cross-attention 读出后，基线由 3% 升至
>   52%，原先报告的 71 个百分点领先中有 49 个来自读出方式而非建模方式。
> — 扩散基线失效：cosine 调度使 `ᾱ` 末项降至 2.4e-7，DDIM 自该项起步导致噪声
>   预测误差放大约 2000 倍，全部配置成功率精确锁死于 2.0%；施加 x₀ 裁剪后恢复
>   正常步数曲线（1→100 步：2/2/3/10/25/33/33%）。
> — 多任务观测空间：将三任务的目标统一为「物体最终位置」并按任务取自各自
>   actor，使多任务由 3/3/0% 提升至 60/96/36%。

**3. 提出并实现「条件动作分布模态检验」作为训练前判据**

> 生成式动作头的理论依据是演示的条件动作分布多模态、回归会塌到非法的条件均值。
> 该前提可在训练前量化：在真值状态空间中检索**跨轨迹**近邻，将其动作块投影至
> 主方向后计算双峰系数。实测三任务为 0.49 / 0.39 / 0.48，均低于正态参考值
> 0.555，即分布为宽单峰、前提不成立。该判据耗时数分钟，可替代一轮 40 分钟训练，
> 并与另外两条判据（动作表示信噪比、冻结特征线性探针的观测充分性）构成完整的
> 训练前检查流程。
>
> 交叉验证：开环动作复现误差显示 Flow Matching 估计条件均值的误差为回归基线的
> 1.5 倍（0.229 对 0.154，归一化动作单位）；将其容量扩大 2.2 倍后训练 loss 下降
> （0.1147→0.1134）而成功率由 60/96/36% 崩至 4/95/8%——更强的生成模型给出的是
> 更忠实的**样本**，而单峰数据所需的是均值。

**4. 语言条件有效性的量化**

> 设计指令交换消融：喂入错误任务的指令时 StackCube 成功率由 32% 降至 0%，
> 而目标区域在相机中可见的 PushCube 不受影响（94–100%）。该结论与冻结特征
> 线性探针独立印证：PushCube 目标的可恢复性 R²=0.99，PickCube 目标 R²≈0
> （每局随机的不可见标记）。即**目标视觉可见时语言冗余，不可见时语言承重**。

---

## 工程与基础设施（可并入某一条，或用于回答「工程能力」类提问）

> 冻结视觉骨干占约 88% 训练时间且管线无图像增强，据此预计算并缓存 DINOv2 patch
> token（与像素路径相对误差 2.2e-4，低于训练所用 bf16 精度），训练吞吐由
> 8.3 提升至 25–36 it/s，使多 seed 与消融矩阵可行；训练支持含优化器状态的
> 断点续跑，实验以阶段级 `.done` 标记的串行队列调度，意外中断最多损失 1000 步。

---

## English

**Summary (≈50 words)**

> Language-conditioned multi-task Flow Matching manipulation policy on ManiSkill3
> (frozen DINOv2 + SigLIP, DiT-style action head, 6.8M trainable parameters). Under a
> matched-encoder, matched-parameter, matched-budget comparison it reaches 60/96/36%
> across three tasks against Diffusion Policy's 44/92/3%, with 10x fewer sampling
> steps. Traced its shortfall against a matched-capacity regression baseline to the
> demonstrations' conditional action distribution being unimodal, and turned that into
> a minutes-long pre-training check.

**Bullets**

> - Implemented a language-conditioned multi-task manipulation policy; 60/96/36% over
>   three ManiSkill3 tasks vs 44/92/3% for a same-backbone Diffusion Policy, at a tenth
>   of the sampling steps (74% at two Euler steps vs 33% at fifty DDIM steps single-task).
> - Found and fixed three flaws that invalidated the comparison: a mean-pooling readout
>   that diluted the goal token to 1/83 (baseline 3% → 52%, accounting for 49 of the 71
>   points previously claimed), a dead diffusion baseline pinned at exactly 2.0% across
>   all configurations from a cosine schedule leaving ᾱ at 2.4e-7, and a non-unified
>   multi-task observation space (3/3/0% → 60/96/36%).
> - Proposed and implemented a **conditional-action-distribution modality check** as a
>   pre-training criterion: cross-trajectory nearest neighbours in ground-truth state
>   space, projected onto their principal direction, yield bimodality coefficients of
>   0.49/0.39/0.48 against a normal reference of 0.555 — the premise for a generative
>   action head does not hold on this data. Minutes to run; replaces a 40-minute run.
> - Quantified language grounding by instruction swapping: success on StackCube falls
>   from 32% to 0% under the wrong instruction, while PushCube (whose goal is visible
>   to the camera, probe R²=0.99) is unaffected at 94–100%.

---

## 面试问答准备

**「你的方法比基线好在哪？」**

> 比扩散基线好，同成功率下采样步数少一个数量级。比回归基线，单任务好、
> 多任务不好——这一点我追下去了，结论写在项目里。

**「多任务上输给行为克隆，你怎么解释？」**

> 依次排除：CFG dropout 的训练条件不对称（有影响，不足以翻盘）、采样步数不足
> （2 到 20 步曲线是平的）、采样过度发散（模型散度 0.325 对数据 0.42，校准良好）、
> 推理时抽样本而非取均值（取 16 个样本平均反而更差）。然后做开环测量，
> 发现流匹配估计条件均值的误差是回归的 1.5 倍。最后去量数据，
> 条件动作分布是单峰的——选生成式动作头的前提不成立。
> 交叉验证是把模型加宽两倍：loss 更低，成功率从 60% 崩到 4%。

**「这个项目最难的部分？」**

> 不是训练，是让对比成立。我最初报过 71 个百分点的领先，其中 49 个后来被证明
> 来自基线的读出方式。现在仓库里有三条训练前判据，每一条都是从一次浪费掉的
> 训练里提炼出来的。

**「哪里没做成？」**

> 三条。没有留出验证集，所以开环误差是训练集拟合，不能当泛化证据；
> 主要结论只有一到两个 seed，而测量噪声实测 3 个百分点；
> 想构造真正多模态的演示来验证「生成式动作头在多模态数据上更强」这条推论，
> 混合运动规划与 PPO 两种解法后条件散度到 1.05，但双峰系数只有 0.547，
> 仍是宽的平顶单峰，所以这条推论**没有**被证实，只证实了它的前提在单峰数据上
> 不成立。
