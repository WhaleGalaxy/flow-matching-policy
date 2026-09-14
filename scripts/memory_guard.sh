#!/usr/bin/env bash
# 内存守护：可用内存连续跌破阈值就停掉实验队列，保住用户的桌面会话。
#
# 存在的理由：2026-09-13 凌晨，三任务特征缓存（15.3GB）把内存顶满，内核回收
# 匿名内存时杀掉了用户的 VS Code。队列本身有阶段标记和断点续跑，被停掉最多
# 损失一千步（约一分钟）；而编辑器被杀掉损失的是用户未保存的工作。
# 两害相权，宁可停队列。
#
# 不用 pgrep 找队列 —— `pgrep -f "queue.sh"` 会匹配到命令行里含这个字符串的
# 任何进程，包括守护自己。一律从 pidfile 读 PID。
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${RUN_DIR:-outputs/$(date +%Y-%m-%d)}"
THRESH_MB="${THRESH_MB:-900}"
STRIKES="${STRIKES:-3}"          # 连续几次跌破才动手，避开瞬时抖动
INTERVAL="${INTERVAL:-30}"
LOG="$RUN_DIR/memory_guard.log"

bad=0
echo "[$(date +%H:%M)] 内存守护启动：阈值 ${THRESH_MB}MB，连续 ${STRIKES} 次触发" >> "$LOG"
while true; do
    qpid=$(cat "$RUN_DIR/queue.pid" 2>/dev/null)
    if [ -z "$qpid" ] || ! kill -0 "$qpid" 2>/dev/null; then
        echo "[$(date +%H:%M)] 队列已结束，守护退出" >> "$LOG"; exit 0
    fi
    avail=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$avail" -lt "$THRESH_MB" ]; then
        bad=$((bad+1))
        echo "[$(date +%H:%M)] 可用内存 ${avail}MB，第 ${bad}/${STRIKES} 次" >> "$LOG"
    else
        bad=0
    fi
    if [ "$bad" -ge "$STRIKES" ]; then
        echo "[$(date +%H:%M)] 触发：停掉队列 PID $qpid 及其训练子进程" >> "$LOG"
        # 先停训练（它才是吃内存的），再停队列，避免队列立刻拉起下一个阶段
        pkill -TERM -P "$qpid" 2>/dev/null
        kill -TERM "$qpid" 2>/dev/null
        sleep 10
        pkill -TERM -f "python -m src.train" 2>/dev/null
        echo "[$(date +%H:%M)] 已停止。重跑 scripts/queue.sh 会跳过已完成阶段并从断点续上。" >> "$LOG"
        exit 1
    fi
    sleep "$INTERVAL"
done
