# 收尾队列 + 补齐 Diffusion 基线的种子（2026-09-14 00:25 编排）。
#
# 为什么要有下半段：主表的三条臂里，FM 有 42/123/2024 三个种子、BC 有 42/123/7
# 三个种子，而 DDPM 只有 seed 42 一个。原因是 outputs/2026-09-12-heldout/ 里有
# 四个 0 字节的 .done 标记（train/eval_ddpm_mt_val10_s{123,7}，时间戳都是
# 09-12 20:34），把这两个种子静默跳过了 —— 没有日志、没有产物、CSV 里也没有行。
# 那些标记已删。
#
# 这是整个项目最容易被一句话问穿的位置：自己的方法三个种子、对比的基线一个种子。
# PickCube 的种子方差是 ±28 个百分点（FM 13/22/65，BC 28/35/79），没有误差棒的
# 单点比较说明不了任何事。
#
# 配置与已跑的 ddpm_mt_val10_s42 逐项对齐，只有 num_workers 从 3 降到 2 ——
# 那是 09-13 内存爆掉后的对策，不影响结果，只影响速度和内存占用。

# ---------- 上半段：原收尾队列 ----------
source scripts/queue_plan_tail.sh

# ---------- 下半段：补 DDPM 基线的 seed 123 与 seed 7 ----------
MT="[PickCube-v1,PushCube-v1,StackCube-v1]"
CACHE=/home/xjy/.maniskill/feat_cache
COMMON="++eval.every=1000000 ++eval.n_episodes=0 ++wandb.enabled=false
        ++feature_cache=$CACHE ++num_workers=2
        ++val_frac=0.1 ++split_seed=0 ++val_every=2500"
MTC="tasks=$MT ++task_goal=true ++steps=60000 ++model.cfg_dropout=0 $COMMON"
CSV="$RUN_DIR/multitask.csv"
COOLDOWN="${COOLDOWN:-60}"

seed_run() {
    local run="$1"; shift
    # 调用方 override 排在 $MTC 之后：Hydra 重复 key 取最后一个。
    # 见 queue_plan_capacity.sh 里同一个坑的记录。
    stage "train_$run" python -m src.train ++run_name="$run" $MTC "$@"
    sleep "$COOLDOWN"
    stage "eval_$run" python scripts/run_eval.py --ckpt "outputs/$run/ckpt_60000.pt" \
        --no-ema --n-episodes 100 --out "$CSV"
    sleep "$COOLDOWN"
}

for s in 123 7; do
    seed_run "ddpm_mt_val10_s$s" model=ddpm ++seed=$s
done

# ---------- 补上两个新 checkpoint 的 100 步重测 ----------
# 这两个阶段在容量队列里跑过并失败了（当时 checkpoint 还不存在），失败不写 .done，
# 所以这里直接重跑即可，与 reference_100steps.csv 里已有的行拼成完整的一张表。
REFCSV="$RUN_DIR/reference_100steps.csv"
for s in 123 7; do
    stage "ref100_ddpm_mt_val10_s$s" python scripts/run_eval.py \
        --ckpt "outputs/ddpm_mt_val10_s$s/ckpt_60000.pt" --no-ema --n-episodes 100 \
        --max-steps 100 --out "$REFCSV"
    sleep "$COOLDOWN"
done

# ---------- 录演示动图 ----------
# 用 bcx_mt_val10_s42：它是三个任务上成绩最好的多任务策略（79/92/50），
# --max-tries 要在每个任务上找到一局成功的 rollout，用 StackCube 只有 18% 的
# Flow Matching 大概率录不出成功画面。
stage record_demo python scripts/record_demo.py \
    --ckpt outputs/bcx_mt_val10_s42/ckpt_60000.pt \
    --out docs/figures/demo.gif

# ---------- 最终报告：三条臂都有三个种子之后重出一遍 ----------
stage report_final python scripts/make_report.py --csv "$CSV" \
    --out-dir docs/figures --weights online
stage report_final_ref python scripts/make_report.py --csv "$REFCSV" \
    --out-dir docs/figures --weights online
