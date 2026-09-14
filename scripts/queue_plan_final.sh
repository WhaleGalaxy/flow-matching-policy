# 最终队列（2026-09-14 01:05 编排），接在 queue_plan_tail_seeds.sh 之后。
#
# 补一个 ref100 的漏网之鱼：那一节的种子循环写的是 42/123/7，而 seed 7 那轮 FM
# 训练发散、权重不可用（outputs/fm_mt_val10_s7/DIVERGED.txt），FM 实际可用的种子
# 集合是 42/123/2024。于是 100 步重测里 FM 只有两个有效种子。
#
# 发散行本身已由 make_report.drop_diverged 自动剔除（判据是 run 目录里的
# DIVERGED.txt），不再依赖手工删 CSV —— 这次复发正是因为原来靠手工。

REFCSV="$RUN_DIR/reference_100steps.csv"
CSV="$RUN_DIR/multitask.csv"
COOLDOWN="${COOLDOWN:-60}"

stage ref100_fm_mt_val10_s2024 python scripts/run_eval.py \
    --ckpt outputs/fm_mt_val10_s2024/ckpt_60000.pt --no-ema --n-episodes 100 \
    --max-steps 100 --out "$REFCSV"
sleep "$COOLDOWN"

# 全部数据齐了之后重出一遍报告，并把主结果表写进 README 的标记区间。
# README 第 44 行那句"两者的数字必须一致，不一致就说明有一边过期了"由此闭合。
stage report_all_multitask python scripts/make_report.py --csv "$CSV" \
    --out-dir docs/figures --weights online --write-readme
stage report_all_ref python scripts/make_report.py --csv "$REFCSV" \
    --out-dir docs/figures --weights online
