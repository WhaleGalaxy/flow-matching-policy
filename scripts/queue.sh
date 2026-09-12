#!/usr/bin/env bash
# 可断点续跑的实验队列。
#
# 为什么要有这个脚本：9-10 的那一轮是直接在交互 shell 里跑的，机器 12:49 关机，
# 12:44 才起步的 DDPM 重测连同后面的分析全部丢掉，日志停在"加载权重"那一行。
# 三条对策都在这里：
#   1. 每个阶段一个 .done 标记，重跑时已完成的阶段直接跳过（关机/崩溃只损失当前阶段）
#   2. 训练本身也能续跑（src/train.py 会自动读 last.pt），单个阶段最多损失 500 步
#   3. 用法见下：setsid + nohup 脱离终端，systemd-inhibit 挡掉待机/合盖休眠
#
#     setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch \
#         --why="fm_policy 实验队列" bash scripts/queue.sh > outputs/<date>/queue.log 2>&1 &
#
# 不用 set -e：一个阶段失败不该带走整条队列，后面的阶段大多互相独立。
set -uo pipefail

cd "$(dirname "$0")/.."
RUN_DIR="${RUN_DIR:-outputs/$(date +%Y-%m-%d)}"
mkdir -p "$RUN_DIR"

stage() {
    local name="$1"; shift
    if [ -f "$RUN_DIR/$name.done" ]; then
        echo "[$(date +%H:%M)] [跳过] $name（已完成）"
        return 0
    fi
    echo "[$(date +%H:%M)] ===== $name ====="
    local t0=$SECONDS
    if "$@" >> "$RUN_DIR/$name.log" 2>&1; then
        touch "$RUN_DIR/$name.done"
        echo "[$(date +%H:%M)] [完成] $name  用时 $(( (SECONDS-t0)/60 )) 分钟"
    else
        echo "[$(date +%H:%M)] [失败] $name  退出码 $?  见 $RUN_DIR/$name.log"
    fi
}

# pidfile 而不是 pgrep：后继队列要判断"前一个队列跑完没有"，而
# `pgrep -f "bash scripts/queue.sh"` 会匹配到任何命令行里**包含**这个字符串的进程
# —— 包括交互 shell 里刚敲下的那条命令本身。踩过一次，表现是后继队列永远等下去。
PIDFILE="$RUN_DIR/queue.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "队列开始：$(date)  RUN_DIR=$RUN_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate robot
export HYDRA_FULL_ERROR=1

# shellcheck disable=SC1090
source "${QUEUE_PLAN:-scripts/queue_plan.sh}"

echo "队列结束：$(date)"
