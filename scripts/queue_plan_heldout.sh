# 留出集 + 多 seed 的重跑计划（2026-09-12）。
#
# 用法：
#   RUN_DIR=outputs/2026-09-12-heldout setsid nohup \
#       env QUEUE_PLAN=scripts/queue_plan_heldout.sh \
#       systemd-inhibit --what=idle:sleep:handle-lid-switch bash scripts/queue.sh &
#
# 为什么要整批重跑，而不是在已有 checkpoint 上补测：
#
# 1. **留出集改变了训练数据。** val_frac=0.1 之后训练集少 10%，所有数字都要重测。
#    旧结果（val_frac=0）不是"同一个实验的另一次测量"，两者不能放进同一张表平均，
#    这一点已经钉进 CSV 的 schema 和 make_report 的分组键。
#
# 2. **主要结论只有 1 个 seed。** 成功率的测量噪声实测约 3 个百分点，
#    多任务 PushCube 上 FM 96% 与 BC 94% 这种差距本来就不成立。有了特征缓存
#    一轮只要 39 分钟，3 个 seed 是现实的。
#
# 3. **旧的三方对比里 cfg_dropout 不对齐。** README 头条那一行 FM 是
#    fmnocfg_mt_tg60_s42（cfg_dropout=0.0），而同一张表里的 DDPM 是
#    ddpm_mt_tg60_s42（cfg_dropout=0.2）。评测一律 guidance=1.0 不走 CFG，
#    所以这个旋钮只在训练时起作用，但它确实是两者之间一个没对齐的变量 ——
#    而"FM 稳赢 Diffusion"是本项目最有信心的一条结论，它不该带着这个瑕疵。
#    这一轮三个方法一律 cfg_dropout=0（BCXAttnPolicy 本来就不用这个旋钮）。
#
# 4. **单任务与多任务的观测路径统一成 task_goal。** 两者对 PickCube 数值等价
#    （一个取 obs/extra/goal_pos，一个取 env_states/actors/goal_site，已对拍），
#    但"单任务与多任务结论方向相反"是本项目的核心主张，能少一个差异就少一个。
#
# 划分由 split_seed=0 决定，与训练 seed 无关，所以所有方法、所有 seed 拿到的是
# 逐条相同的训练集与留出集。见 src/data/dataset.split_episodes。

MT="[PickCube-v1,PushCube-v1,StackCube-v1]"
CACHE=/home/xjy/.maniskill/feat_cache
HELD="++val_frac=0.1 ++split_seed=0 ++val_every=2500"
COMMON="++eval.every=1000000 ++eval.n_episodes=0 ++wandb.enabled=false
        ++feature_cache=$CACHE ++num_workers=3 $HELD"
# 三个方法共用：每任务取目标、60000 步（每个任务分到约 20000 步）、不做指令置空
MTC="tasks=$MT ++task_goal=true ++steps=60000 ++model.cfg_dropout=0 $COMMON"
PICKC="tasks=[PickCube-v1] ++task_goal=true ++steps=20000 ++model.cfg_dropout=0 $COMMON"

CSV="$RUN_DIR/multitask.csv"
PICKCSV="$RUN_DIR/pick.csv"

mt_run() {          # mt_run <run 名> <评测 episode 数> <额外 train 参数...>
    local run="$1" nep="$2"; shift 2
    stage "train_$run" python -m src.train ++run_name="$run" "$@" $MTC
    stage "eval_$run"  python scripts/run_eval.py --ckpt "outputs/$run/ckpt_60000.pt" \
        --no-ema --n-episodes "$nep" --out "$CSV"
}
pick_run() {
    local run="$1" nep="$2"; shift 2
    stage "train_$run" python -m src.train ++run_name="$run" "$@" $PICKC
    stage "eval_$run"  python scripts/run_eval.py --ckpt "outputs/$run/ckpt_20000.pt" \
        --no-ema --n-episodes "$nep" --out "$PICKCSV"
}
report()  { stage "report_$1" python scripts/make_report.py --csv "$CSV" \
        --out-dir docs/figures --weights online; }

# ---------- A：多任务三方对比 × 3 个 seed ----------
# 按 seed 分块而不是按方法分块：任何时刻中断，手上都是一张**完整**的三方对比表，
# 只是 seed 少一些。按方法分块的话中断就会剩下"FM 有三个 seed、BC 一个都没有"。
for s in 42 123 7; do
    mt_run "fm_mt_val10_s$s"   100 ++seed=$s
    mt_run "ddpm_mt_val10_s$s" 100 model=ddpm     ++seed=$s
    mt_run "bcx_mt_val10_s$s"  100 model=bc_xattn ++seed=$s
    report "mt_s$s"
done

# ---------- B：单任务 PickCube × 3 个 seed ----------
# 这张表要回答的是"单任务下 FM 领先，多任务下反过来"，所以它必须与 A 用
# 同一个划分、同一个观测路径、同一个 cfg_dropout。
for s in 42 123 7; do
    pick_run "fm_pick_val10_s$s"   100 ++seed=$s
    pick_run "ddpm_pick_val10_s$s" 100 model=ddpm     ++seed=$s
    pick_run "bcx_pick_val10_s$s"  100 model=bc_xattn ++seed=$s
done
# 平均池化读出的 BC 只是反面对照（读出方式做错了会怎样），一个 seed 足够：
# 它要支撑的结论是"3% 与 52% 差着数量级"，不是一个需要误差棒的差距。
pick_run "bc_pick_val10_s42" 100 model=bc ++seed=42
stage report_pick python scripts/make_report.py --csv "$PICKCSV" \
    --out-dir docs/figures --weights online

# ---------- C：容量消融 × 3 个 seed ----------
# "loss 更低、成功率崩掉"是本项目最反直觉的一条结论，而它现在只有 1 个 seed。
# 加了留出集之后还多一件事可做：看 val loss 曲线是不是掉头向上 ——
# 那会把"过拟合"从推断变成直接观测。
for s in 42 123 7; do
    mt_run "fmwide_mt_val10_s$s" 100 ++model.d_model=384 ++seed=$s
done
report wide

# ---------- D：依赖新 checkpoint 的分析 ----------
# 动作复现误差现在算在留出集上，报的是**泛化**误差而不是拟合误差。
stage action_accuracy python scripts/action_accuracy.py \
    --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt outputs/bcx_mt_val10_s42/ckpt_60000.pt \
    --task PickCube-v1 --split val
stage action_accuracy_train python scripts/action_accuracy.py \
    --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt outputs/bcx_mt_val10_s42/ckpt_60000.pt \
    --task PickCube-v1 --split train
stage language_fm python scripts/language_ablation.py \
    --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt --n-episodes 50
stage language_bcx python scripts/language_ablation.py \
    --ckpt outputs/bcx_mt_val10_s42/ckpt_60000.pt --n-episodes 50 \
    --out-json outputs/language_ablation_bcx_val10.json \
    --fig docs/figures/language_ablation_bcx_val10.png
stage failure_modes python scripts/failure_modes.py \
    --ckpt outputs/fm_pick_val10_s42/ckpt_20000.pt \
           outputs/ddpm_pick_val10_s42/ckpt_20000.pt \
           outputs/bcx_pick_val10_s42/ckpt_20000.pt \
           outputs/bc_pick_val10_s42/ckpt_20000.pt \
    --labels FM Diffusion "BC cross-attn" "BC mean-pool" --n-episodes 100
