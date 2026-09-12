#!/usr/bin/env bash
# 等上一条队列真正退出，然后跑指定的计划。
# 判据是 pidfile 里的进程还在不在 —— 不能用 `pgrep -f "bash scripts/queue.sh"`，
# 它会匹配到命令行里含这个字符串的任何进程（包括刚敲下的那条命令本身），
# 后果是后继队列永远等下去。这个坑实际踩过一次，静默卡了十分钟。
cd "$(dirname "$0")/.."
RUN_DIR="${RUN_DIR:-outputs/$(date +%Y-%m-%d)}"
PLAN="${1:?用法: wait_then_run.sh <plan.sh>}"
PIDFILE="$RUN_DIR/queue.pid"
while [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; do
    sleep 30
done
exec env RUN_DIR="$RUN_DIR" QUEUE_PLAN="$PLAN" \
     systemd-inhibit --what=idle:sleep:handle-lid-switch --why="fm_policy 队列" \
     bash scripts/queue.sh
