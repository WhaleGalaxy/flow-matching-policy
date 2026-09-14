# 容量曲线（2026-09-12，接在 queue_plan_heldout.sh 之后）。
#
# 要补的是一个**缺失的对照**，不是多一个消融。
#
# 项目里最反直觉的一条结论是：把 FM 从 6.80M 加宽到 15.01M，训练 loss 更低
# （0.1147 → 0.1134）而成功率崩掉（60/96/36 → 4/95/8），解释是"更好的生成模型
# 给的是更忠实的样本，而单峰数据上你要的是均值"。
#
# 但这个解释有一个没被排除的竞争假设：**加宽就是会过拟合，跟生成式动作头无关。**
# 要分开这两者只需要一个对照 —— 把同样的宽度给等条件的回归基线：
#
#   BC 在 384 上也崩  → 是普通过拟合，与建模方式无关，原解释不成立
#   BC 在 384 上不崩  → 代价确实是生成式动作头特有的，原解释成立
#
# 这个对照一直没跑，而它只要 39 分钟。有了留出集之后还多一路证据：
# val loss 掉不掉头，能把"过拟合"从推断变成直接观测。
#
# 顺带把 128 这一档补上，于是 FM 与 BC 各有 128 / 256 / 384 三个点，
# 是一条曲线而不是两个孤立的点。README"已知局限"里
# "'容量'这个变量没有被单独扫过一条曲线"那一条由此闭掉。

MT="[PickCube-v1,PushCube-v1,StackCube-v1]"
CACHE=/home/xjy/.maniskill/feat_cache
# 2026-09-13：内存爆过一次，VS Code 被内核杀掉。根因不是泄漏，是三个任务的
# 特征缓存合计 15.3GB，与整机内存（15.7GB）相当，而 DataLoader 随机采样会把
# 整份缓存都摸进页缓存，内核为了留住它们去回收别的进程的匿名内存。
# 两处对策：worker 从 3 降到 2；整条队列跑在 MemoryMax=11G 的 cgroup 里，
# 于是内存压力只能让它回收自己的页缓存（可从磁盘重读），碰不到用户的会话。
COMMON="++eval.every=1000000 ++eval.n_episodes=0 ++wandb.enabled=false
        ++feature_cache=$CACHE ++num_workers=2
        ++val_frac=0.1 ++split_seed=0 ++val_every=2500"
MTC="tasks=$MT ++task_goal=true ++steps=60000 ++model.cfg_dropout=0 $COMMON"
CSV="$RUN_DIR/multitask.csv"

# 每个阶段之间歇一分钟。实测 GPU 只有 56°C、没有任何热降频，所以这不是必须的；
# CPU 封装 87°C 是这台机器上更热的部件（评测的物理仿真跑在 CPU 上）。
# 二十来个阶段总共多花二十分钟，买个安心。
COOLDOWN="${COOLDOWN:-60}"

cap_run() {
    local run="$1"; shift
    # 调用方的 override 必须排在 $MTC **后面**：Hydra 对重复 key 取最后一个。
    # 踩过一次，代价是两个 seed 白跑：$MTC 里硬编码了 ++model.cfg_dropout=0，
    # 原来的顺序把 cfg02 臂传进来的 0.2 又改回 0，于是那条臂训出来的是
    # ddpm_mt_val10_s42 的复制品，而 run_name 还叫 cfg02。
    # 判定方法：训练日志开头会打印解析后的配置，以那里的值为准，别看命令行。
    stage "train_$run" python -m src.train ++run_name="$run" $MTC "$@"
    sleep "$COOLDOWN"
    stage "eval_$run"  python scripts/run_eval.py --ckpt "outputs/$run/ckpt_60000.pt" \
        --no-ema --n-episodes 100 --out "$CSV"
    sleep "$COOLDOWN"
}

# ---------- 失败归因：同一个方法的两个种子，差距出在动作链条的哪一段 ----------
#
# FM 的两个种子在 PickCube 上是 65% 和 13%，而两者的验证 loss 是 0.1510 和 0.1686，
# 几乎一样；评测用的还是完全相同的 100 个初始场景。一个成功率数字说不出这 52 个
# 百分点丢在哪里，而 PickCube 的链条（接近 → 抓取 → 送达 → 静止）是环境自己
# 暴露出来的，可以逐段判定。
stage failure_seed_gap python scripts/failure_modes.py \
    --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt outputs/fm_mt_val10_s123/ckpt_60000.pt \
    --labels "FM seed42" "FM seed123" --n-episodes 100

# ---------- 补一个替代种子：seed 7 那轮 FM 发散了 ----------
#
# fm_mt_val10_s7 从第 35400 步开始渐进失稳，梯度范数爬到 1.1e6，验证 loss 从
# 0.158 涨到 0.713。那一轮的评测记录已从结果表剔除（见 outputs/fm_mt_val10_s7/
# DIVERGED.txt），所以 FM 只剩两个可用种子，这里补第三个。
#
# 用 2024 而不是重跑 7：如果失稳是那个初始化特有的，重跑只会再废一小时。
# 代价是 FM 的种子集合（42/123/2024）与 BC 的（42/123/7）不同 —— 种子之间可交换，
# 不影响 mean±std 的含义，但这件事要写进结果说明里。
#
# src/train.py 现在有发散看门狗（验证 loss 连续两次高出历史最低 1.5 倍即中止），
# 所以再遇到同样的事最多浪费十几分钟，不会再是一小时。
cap_run fm_mt_val10_s2024 ++seed=2024

# ---------- 最优先：把 DDPM 的 cfg_dropout 放回它自己的工作点 ----------
#
# queue_plan_heldout.sh 里我把三个方法的 cfg_dropout 统一成 0，理由是
# "同骨干同预算"那句话里不该留一个没对齐的变量。seed 42 跑完发现这个决定可能是
# 错的：
#
#   旧 DDPM（cfg_dropout=0.2，全量数据）  44 / 92 / 3
#   新 DDPM（cfg_dropout=0，留出 10%）     8 / 96 / 1
#
# 而同样的数据削减下 FM 是稳的（60 → 65）。也就是说"对齐"很可能把 Diffusion
# 基线打残了 —— 这正是本项目最不能犯的错误：基线弱所以自己的方法看起来赢。
#
# 判断错在哪：cfg_dropout 是**训练超参**，不是观测配置。本项目对采样步数早就
# 采取了"各方法用自己的默认工作点"（FM 10 步、DDPM 100 步），cfg_dropout 应当
# 同理。强行统一不是更公平，是让两个方法都不在自己的工作点上。
#
# 所以补一条 cfg_dropout=0.2 的 DDPM 臂，先跑 seed 42 看假设成不成立。
#
# 2026-09-13 22:00 结果（全部对齐到 seed 42、同一套 100 个初始场景）：
#
#   全量数据   cfg_dropout=0.2   44 / 92 / 3   （ddpm_mt_tg60_s42）
#   留出 10%   cfg_dropout=0      8 / 96 / 1   （ddpm_mt_val10_s42）
#   留出 10%   cfg_dropout=0.2    7 / 98 / 0   （本臂）
#
# 假设被证伪：把 cfg_dropout 放回它自己的工作点，PickCube 还是 7%。它与 0 的
# 8% 落在同一噪声里 —— 同配置重跑一次是 4/95/0，这就是这套评测的重复噪声。
# 44 → 7 的崩塌跟着的是那 10% 留出，不是 cfg_dropout；而 FM 在同样的削减下是
# 63 → 65。结论因此不是"我们把基线打残了"，而是"Diffusion 基线对这个数据削减
# 异常敏感，FM 不敏感"。这一条要写进结果说明。
#
# 只跑 seed 42，不跑 123 / 7：这条对照是**配对**的（同种子、同划分、同评测场景），
# 要测的又是 36 个百分点的大效应，实测差 1 个点。再花 2.5 小时补两个种子，买到的
# 是一个零结果的误差棒，而主表报的是 cfg_dropout=0 那条臂（已有三个种子），
# 这里只是消融脚注。省下的机时给后面那条决定性的对照和外部口径重测。
cap_run ddpm_mt_val10_cfg02_s42 model=ddpm ++model.cfg_dropout=0.2 ++seed=42
stage report_ddpm_cfg python scripts/make_report.py --csv "$CSV" \
    --out-dir docs/figures --weights online

# ---------- 决定性的那一个：加宽的回归基线 ----------
# 一个 seed 够用：这条对照要回答的是"加宽之后 BC 崩不崩"，
# 是定性问题（崩到个位数 vs 保持在几十），不需要误差棒。
cap_run bcx_mt_val10_d384_s42 model=bc_xattn ++model.d_model=384 ++seed=42
stage report_cap_a python scripts/make_report.py --csv "$CSV" \
    --out-dir docs/figures --weights online

# 128 那一档（凑成一条完整容量曲线）暂时砍掉，为了把机器早点空出来。
# 要补的话把下面两行取消注释即可：
#   cap_run fm_mt_val10_d128_s42  ++model.d_model=128 ++seed=42
#   cap_run bcx_mt_val10_d128_s42 model=bc_xattn ++model.d_model=128 ++seed=42

# ---------- 外部参照口径：把评测预算对齐到官方基线的 100 步 ----------
#
# ManiSkill3 论文附录 Table III 给了同样四个任务、同样的运动规划演示、
# 同样单个 128×128 base_camera 下 BC / ACT / Diffusion Policy 的 RGB 成绩。
# 那是本项目唯一现成的外部参照点，但协议有一处差得很远：
#
#   ManiSkill 给 PickCube-v1 注册的 max_episode_steps 是 50
#   官方基线脚本用 --max_episode_steps 100
#   本项目的 rollout 用 300，而且 `truncated` 收下了却从没用过
#
# 三者的成功率不可比，而且差异对本项目**有利**。内部对比不受影响（所有方法
# 拿到同一个预算），但任何与已发表数字的比较都必须换到同一个预算上。
# 所以这里用 100 步把主结果重测一遍，写进单独的 CSV。
# 300 步那一份保留，两者的差值本身有信息：它量的是"再给三倍时间能多救回多少"。

REFCSV="$RUN_DIR/reference_100steps.csv"
ref_eval() {   # ref_eval <run 名> <ckpt 步数> <csv>
    stage "ref100_$1" python scripts/run_eval.py \
        --ckpt "outputs/$1/ckpt_$2.pt" --no-ema --n-episodes 100 \
        --max-steps 100 --out "$3"
    sleep "$COOLDOWN"
}
for s in 42 123 7; do
    for m in fm ddpm bcx; do
        ref_eval "${m}_mt_val10_s$s"   60000 "$REFCSV"
        ref_eval "${m}_pick_val10_s$s" 20000 "$REFCSV"
    done
done
stage report_ref python scripts/make_report.py --csv "$REFCSV" \
    --out-dir docs/figures --weights online
