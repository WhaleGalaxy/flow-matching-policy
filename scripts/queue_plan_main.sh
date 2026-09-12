# 主实验计划（多任务受控对比 + 消融）。
#
# 用法：
#   RUN_DIR=outputs/$(date +%F) setsid nohup env QUEUE_PLAN=scripts/queue_plan_main.sh \
#       systemd-inhibit --what=idle:sleep:handle-lid-switch bash scripts/queue.sh &
#
# 原为 2026-09-11 那一轮的阶段 G：目标语义统一之后的多任务受控对比 + 全部消融。
#
# 与阶段 C/E 的区别只有一个：++task_goal=true。每个任务从自己的 actor 取目标位置
# （PickCube 是不可见的目标标记，PushCube 是桌面目标区域，StackCube 是下面那块
# 方块），proprio 恒为 12 维，与单任务 use_goal 的观测空间完全一致。
#
# 之前两版失败的记录留在 multitask.csv 里，它们是这条路线的消融：
#   fm_mt_s42       goal_slot（13 维 + 有效位）        20000 步   3 / 82 / 4
#   fmgc_mt_s42     goal_slot + AdaLN 全局条件         20000 步   6 / 79 / 2
#   fm_mt_tg_s42    每任务目标（12 维）                20000 步  14 / 85 / 2
# 前两行说明条件**注入位置**不是瓶颈，条件**内容**才是。第三行说明内容修对之后
# 还差预算：20000 步分摊给三个任务，每个只拿到约 6700 步，而单任务 PickCube
# 在 8000 步才 37%。所以这一版给 60000 步，让每个任务拿到与单任务相同的 20000 步。
#
# 每个大块之后重出一次结果表，任何时刻中断都有当前最新的表。

MT="[PickCube-v1,PushCube-v1,StackCube-v1]"
CACHE=/home/xjy/.maniskill/feat_cache
COMMON="++steps=60000 ++eval.every=60000 ++eval.n_episodes=0 ++wandb.enabled=false
        ++feature_cache=$CACHE ++num_workers=3"
MTC="tasks=$MT ++task_goal=true $COMMON"
CSV="$RUN_DIR/multitask.csv"
PICKCSV="$RUN_DIR/sweeps.csv"

FM=outputs/fmgoal_PickCube_s42/ckpt_20000.pt
DDPM=outputs/ddpmgoal_PickCube_s42/ckpt_20000.pt
BCPOOL=outputs/bcgoal_PickCube_s42/ckpt_20000.pt
BCX=outputs/bcx_PickCube_s42/ckpt_20000.pt

train_eval() {
    local run="$1"; shift
    stage "g_train_$run" python -m src.train ++run_name="$run" "$@" $MTC
    stage "g_eval_$run"  python scripts/run_eval.py --ckpt "outputs/$run/ckpt_60000.pt" \
        --no-ema --n-episodes 100 --out "$CSV"
}
report() { stage "g_report_$1" python scripts/make_report.py --csv "$CSV" \
        --out-dir docs/figures --weights online; }

# ---------- F1：决定性的一轮 ----------
train_eval fm_mt_tg60_s42 ++seed=42
report a

# ---------- F2：多任务三方对比 ----------
train_eval ddpm_mt_tg60_s42 model=ddpm     ++seed=42
train_eval bcx_mt_tg60_s42  model=bc_xattn ++seed=42
report b

# ---------- F3：语言到底有没有被用上（不用重训） ----------
stage g3_language_fm python scripts/language_ablation.py \
    --ckpt outputs/fm_mt_tg60_s42/ckpt_60000.pt --n-episodes 50

# ---------- F4：观测充分性探针做一次预测检验 ----------
# PickCube 的目标探针早就测出 R²≈0（不可见）。PushCube 的目标区域画在桌面上，
# 而 ManiSkill 在视觉观测模式下**故意不给** goal_pos —— 环境设计者认为它该被看见。
# 同一个探针、同样的留出集协议，在它没见过的任务上做预测。
for t in PickCube-v1 PushCube-v1 StackCube-v1; do
    stage "g4_probe_${t%%-*}" python scripts/probe_observability.py --task "$t" --n-samples 600
done

# ---------- F5：单任务的失败归因（四个策略，含修好的 DDPM 与新的 BC 对照） ----------
stage g5_failure_modes python scripts/failure_modes.py \
    --ckpt "$FM" "$DDPM" "$BCX" "$BCPOOL" \
    --labels FM Diffusion "BC cross-attn" "BC mean-pool" --n-episodes 100

# ---------- F6：拆混淆 + FM 侧改进 + 第二个 seed ----------
# 顺序按价值排：OT-CFM 是 FM 侧唯一的方法改进（闭掉 README"已知局限"里那条
# "使用独立耦合而非 minibatch OT 耦合"），第二个 seed 决定差距算不算噪声。
# 多任务的平均池化 BC 去掉了 —— 读出方式的混淆在单任务上已经拆干净（52% vs 3%），
# 在多任务上重复一遍要多花 40 分钟，买不到新结论。
train_eval fmot_mt_tg60_s42 ++model.ot_coupling=true ++model.time_dist=logitnormal ++seed=42
train_eval fm_mt_tg60_s123  ++seed=123
report c

# ---------- F7：少步区扫描 ----------
stage g7_sweep_fm   python scripts/run_eval.py --ckpt outputs/fm_mt_tg60_s42/ckpt_60000.pt \
    --no-ema --n-episodes 100 --sweep-steps 1 2 4 8 16 --out "$CSV"
stage g7_sweep_fmot python scripts/run_eval.py --ckpt outputs/fmot_mt_tg60_s42/ckpt_60000.pt \
    --no-ema --n-episodes 100 --sweep-steps 1 2 4 8 16 --out "$CSV"
stage g7_sweep_ddpm python scripts/run_eval.py --ckpt outputs/ddpm_mt_tg60_s42/ckpt_60000.pt \
    --no-ema --n-episodes 100 --sweep-steps 1 2 4 8 16 --out "$CSV"
report d

# ---------- F8：其余消融 ----------
train_eval fmnoprop_mt_tg60_s42 ++use_proprio=false ++seed=42
stage g8_language_ddpm python scripts/language_ablation.py \
    --ckpt outputs/ddpm_mt_tg60_s42/ckpt_60000.pt --n-episodes 50 \
    --out-json outputs/language_ablation_ddpm.json \
    --fig docs/figures/language_ablation_ddpm.png
stage g8_action_diversity python scripts/action_diversity.py \
    --fm outputs/fm_mt_tg60_s42/ckpt_60000.pt --bc outputs/bcx_mt_tg60_s42/ckpt_60000.pt \
    --task PickCube-v1 --n-steps 1 2 10 50 100
stage g8_benchmark python scripts/benchmark_inference.py
report e
stage g8_report_pick python scripts/make_report.py --csv "$PICKCSV" \
    --out-dir docs/figures --weights online

# ---------- F9：单任务执行长度扫描与部署可行域（支撑结果，放最后） ----------
stage g9_ddpm_horizon python scripts/run_eval.py --ckpt "$DDPM" --no-ema \
    --n-episodes 100 --sweep-steps 10 --sweep-horizon 1 2 4 16 --out "$PICKCSV"
stage g9_pareto python scripts/pareto.py --csv "$PICKCSV" \
    --ckpt fm="$FM" ddpm="$DDPM" bc="$BCX"
