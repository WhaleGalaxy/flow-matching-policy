#!/usr/bin/env bash
# 把 ManiSkill3 的演示轨迹重放成带 RGB 观测的数据集。
#
# 官方下载的 demo 里 obs/ 是空的（采集时 obs_mode="none"，只存动作和环境状态），
# 必须重放一遍、边跑边渲染，才能得到策略训练需要的图像观测。
# 这一步需要 GPU Vulkan（见 setup_vulkan.sh）。
#
# 产物： <task>/motionplanning/trajectory.rgb.pd_joint_pos.physx_cpu.h5
set -euo pipefail

DEMO_ROOT="${HOME}/.maniskill/demos"
NPROC="${NPROC:-4}"
COUNT="${COUNT:-1000}"

for task in PickCube-v1 StackCube-v1 PegInsertionSide-v1; do
    dir="${DEMO_ROOT}/${task}/motionplanning"
    echo "==> 重放 ${task}（最多 ${COUNT} 条，${NPROC} 进程）"
    ( cd "$dir" && python -m mani_skill.trajectory.replay_trajectory \
        --traj-path trajectory.h5 \
        --save-traj --obs-mode rgb \
        --use-first-env-state \
        --count "$COUNT" -n "$NPROC" 2>&1 | tail -2 )
done

echo
echo "==> 完成，产物："
du -sh "${DEMO_ROOT}"/*/motionplanning/trajectory.rgb.*.h5
