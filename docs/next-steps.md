# 收尾计划：把项目做到可展示

更新于 2026-09-10。目标是**面试可展示**，不是论文完整度。判断标准：
排查能力 > 对比是否成立 > 跑通 > 规模。说了数据不支持的话是唯一的倒扣分项。

## 当前结果（截至 2026-09-10）

100 episode 评测，`outputs/results.csv`。**在线权重**列是可信的那一列，原因见下。

| 配置 | EMA | 在线 |
|---|---|---|
| **FM + goal · PickCube** | 2% | **75%**（平均步长 137 / 上限 300） |
| FM 无 goal · PickCube | 5% | 6% |
| BC 无 goal · PickCube | 4% | 4% |
| DDPM 无 goal · PickCube | 2% | 1% |
| FM 单任务 · StackCube | 0% | 18% |
| FM 单任务 · PegInsertionSide | 0% | 0% |
| FM 多任务 · Pick / Stack / Peg | 3 / 3 / 0% | 3 / 3 / 0% |

延迟基准（独占 GPU，不依赖训练质量）：

| | 延迟 | 去噪部分 | 有效控制频率 |
|---|---|---|---|
| FM 10 步 | 14.31 ms | 12.39 ms | 559 Hz |
| DDPM 100 步 | 132.98 ms | 131.06 ms | 60 Hz |
| | **9.3x** | **10.6x** | |

共有的观测编码开销 1.92 ms（两模型相同）。10.6x 与 10 倍步数比精确对应。

## 这批数字里哪些不能用

1. **FM 75% 与 BC 4% / DDPM 2% 不可比。** 前者观测含 `goal_pos`，后者不含。
   放进同一张表等于在比观测配置，不是建模方式。**这是当前最大的问题。**
2. **消融（采样步数 / CFG）跑在 4% 的 checkpoint 上**，全落在噪声里，无解释力。
3. **BC 不是等容量对照**：1.77M vs FM/DDPM 的 6.80M。等容量需 `model.hidden=2364`。
4. **测量噪声下限 3 个百分点**：同一 checkpoint 同一配置三次评测得到 4% / 5% / 7%。
   任何小于这个尺度的差距都不成立。
5. **训练中途的评测用 EMA 权重，在本项目里近乎无效**：FM+goal 训练日志五个点写的是
   8/4/4/4/4%，而同一批权重的在线版本是 75%。`ema_decay=0.9999` 的时间常数是
   10000 步，对 20000 步的训练太慢。**评测一律用 `--no-ema`。**

## 执行清单

### 0. 先修一个 bug（几分钟）

`scripts/diagnose_rollout.py` 自成一套 proprio 组装（直接用 `qpos`），没跟进
`src/data/dataset.py` 的 `build_proprio`。遇到 `use_goal` 的 checkpoint 会崩：

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x9 and 12x128)

改成调用 `build_proprio`，并加 `--use-goal` 从 checkpoint 的 cfg 读取（照
`scripts/run_eval.py` 的做法）。

### 1. 两个便宜的确认（约 15 分钟）

    # a) 75% 是否还在爬 —— 决定要不要加长到 30000 步
    for s in 12000 16000 20000; do
      python scripts/run_eval.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_$s.pt \
        --n-episodes 100 --no-ema
    done

    # b) EMA 2% vs 在线 75% 的差距过大，确认不是 checkpoint 损坏
    python scripts/diagnose_rollout.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_20000.pt \
      --n-episodes 20              # EMA
    python scripts/diagnose_rollout.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_20000.pt \
      --n-episodes 20 --no-ema     # 在线

判据：(a) 若 16000 → 20000 仍明显上升，把最终配置改到 30000 步重跑；否则维持 20000。
(b) EMA 的抓取率应显著低于在线但不应为零；若为零则另有问题，先查清再往下走。

### 2. 受控对比（约 3 小时，核心交付）

三个方法必须在**完全相同的观测配置**下训练：`use_goal=true`，20000 步（或第 1 步
定下的步数），PickCube 单任务。

    # FM 第二个 seed（第一个 seed 的 checkpoint 已存在）
    python -m src.train ++use_goal=true ++steps=20000 ++seed=123 \
      ++run_name=fmgoal_PickCube_s123

    # BC —— 必须同时补上等容量，否则差距里混着 3.8 倍的参数量
    python -m src.train model=bc model.hidden=2364 ++use_goal=true ++steps=20000 \
      ++run_name=bcgoal_PickCube_s42

    # DDPM
    python -m src.train model=ddpm ++use_goal=true ++steps=20000 \
      ++run_name=ddpmgoal_PickCube_s42

每个训完跑 `python scripts/run_eval.py --ckpt <ckpt> --n-episodes 100 --no-ema`。

时间参考（实测）：FM 42.6 分钟 / BC 37 分钟 / DDPM 52 分钟，100-episode 评测
FM 约 2.5 分钟、DDPM 约 10 分钟。

误差棒：FM 两个 seed 是底线。基线单 seed 可接受，但 README 必须写明并给出
3 个百分点的测量噪声。时间允许的话基线各补一个 seed 更稳。

### 3. 主图：质量 vs 采样步数（约 50 分钟，复用 checkpoint 不重训）

**这是最该花力气的一张图，也是全项目性价比最高的一步。**

    python scripts/run_eval.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_20000.pt \
      --n-episodes 100 --no-ema --sweep-steps 1 2 5 10 20
    python scripts/run_eval.py --ckpt outputs/ddpmgoal_PickCube_s42/ckpt_20000.pt \
      --n-episodes 100 --no-ema --sweep-steps 1 2 5 10 20 50 100
    python scripts/run_eval.py --ckpt outputs/fmgoal_PickCube_s42/ckpt_20000.pt \
      --n-episodes 100 --no-ema --sweep-guidance 1.0 1.5 2.0

期望的图形：FM 在 5–10 步到达平台，DDPM 需要 50–100 步才追上。**如果成立，
项目的主张就从"FM 成功率更高"（大概率站不住）换成"同等质量下推理便宜一个数量级"
（有延迟基准和这张图两条独立证据）** —— 后者恰好也是 Flow Matching 本来的卖点。

若 DDPM 在少步下并没有明显退化，如实呈现，并在 README 里讨论为什么
（动作块维度低、任务相对简单，都是合理解释）。

### 4. README 重写

- **标题去掉或限定 "Multi-Task"**。多任务实测 3/3/0%，撑不起这个词。
- 结果表用 `python scripts/make_report.py --weights online --write-readme`。
- **观测规格必须与数字并列**：`goal_pos` 进入 proprio 是决定性的（6% → 75%），
  不写清楚读者无法判断这个 75% 的含义。它本来就在 ManiSkill 的 state 观测里，
  不是作弊，但必须显式声明。
- 把两轮排查提到显著位置（`docs/debugging.md`），这是仓库最有价值的部分。
- 已知局限如实写：
  - PegInsertion 0%，只用 `base_camera`，演示里的 `hand_camera` 被舍弃了
  - StackCube 单任务 18% / 多任务 3%，多任务预算稀释是真实因素
  - 多任务与 `use_goal` 存在架构冲突：StackCube / PegInsertion 的观测里没有
    `goal_pos`（目标是场景实物），统一观测空间与提供显式目标不可兼得
- 写明评测用在线权重而非 EMA，并给出理由（2% vs 75%）。这与通行做法相反，
  必须有据。

## 明确不做

- **PegInsertion 不救。** 原因清楚（缺腕部相机），投入几小时仍大概率难看。
- **StackCube 不接腕部相机。** 需要双相机编码器，是几小时的改动加训练。
  把"下一步该怎么做、依据是什么"写清楚，面试时讲出来效果接近。
- **不做多 seed 的大规模扫描。** 面试项目不需要，把测量噪声写明更有说服力。

## 验收标准

做完之后，这个仓库应当能回答面试官的这几个问题而不心虚：

1. 你的方法比基线好在哪？—— 同等质量下推理快 9.3 倍，有延迟基准和步数消融两条证据
2. 对比公平吗？—— 同编码器、同 denoiser 结构、同参数量、同训练预算、同观测配置
3. 你怎么知道差距不是噪声？—— 测量噪声实测 3 个百分点，差距大于它，且有多 seed
4. 哪里没做成，为什么？—— 两个任务失败，原因已定位到观测不足，下一步明确
5. 你踩过最难的坑是什么？—— 动作表示信噪比、目标不可观测，都有量化证据和对照组
