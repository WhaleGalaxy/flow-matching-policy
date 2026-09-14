# 收尾队列：补跑因 obs_proprio 那个 bug 崩掉的两个失败归因阶段。
#
# 崩因见 docs/debugging.md：scripts/failure_modes.py 自己抄了一份组装观测的代码，
# 只认识 use_goal 不认识 task_goal，拿 task_goal 训的 checkpoint 去跑就是 9 维
# 对 12 维的 RuntimeError。已改为共用 src.evaluate.obs_proprio。

# 两个阶段必须各写各的文件：failure_modes.py 的 --out-json / --fig 默认值是
# outputs/failure_modes.json 与 docs/figures/failure_modes.png，不传就是两个阶段
# 写同一个路径，后跑的那个把先跑的覆盖掉，只剩最后一份结果。
stage tail_failure_seed_gap python scripts/failure_modes.py \
    --ckpt outputs/fm_mt_val10_s42/ckpt_60000.pt outputs/fm_mt_val10_s123/ckpt_60000.pt \
    --labels "FM seed42" "FM seed123" --n-episodes 100 \
    --out-json outputs/failure_seed_gap.json \
    --fig docs/figures/failure_seed_gap.png

stage tail_failure_modes python scripts/failure_modes.py \
    --ckpt outputs/fm_pick_val10_s42/ckpt_20000.pt \
           outputs/ddpm_pick_val10_s42/ckpt_20000.pt \
           outputs/bcx_pick_val10_s42/ckpt_20000.pt \
           outputs/bc_pick_val10_s42/ckpt_20000.pt \
    --labels FM Diffusion "BC cross-attn" "BC mean-pool" --n-episodes 100 \
    --out-json outputs/failure_modes.json \
    --fig docs/figures/failure_modes.png
