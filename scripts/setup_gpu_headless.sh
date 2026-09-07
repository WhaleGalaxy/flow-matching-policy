#!/usr/bin/env bash
# 恢复 CUDA 计算能力，但完全不碰显示栈。
#
# 背景：完整的 nvidia-driver-570 元包同时装三样东西——
#   ① 内核模块（CUDA 计算，我们要的）
#   ② xserver-xorg-video-nvidia-570（Xorg 显示驱动）
#   ③ libnvidia-gl-570（顶替系统 OpenGL）
# 在 Intel核显 + RTX4060 混合显卡笔记本 + Ubuntu 20.04 的老 GNOME/Xorg 上，
# ②③ 一旦生效 gdm 会切到 NVIDIA 显示路径导致 GUI 起不来。
# headless 版只装 ①，显示继续走 i915/Mesa。
#
# 出问题时的退路（和你上次做的一样）：
#   sudo apt purge '^nvidia-.*' && sudo apt autoremove
#
# 用法： sudo bash setup_gpu_headless.sh
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }

echo "==> [1/4] 安装 headless 计算栈（会一并装回 dkms）"
apt-get update
apt-get install -y nvidia-headless-570 nvidia-utils-570

echo
echo "==> [2/4] 确认没有误装显示相关包"
if dpkg -l | grep -qE "^ii +(xserver-xorg-video-nvidia|libnvidia-gl-)"; then
    echo "!!! 危险：装上了显示相关包，会有黑屏风险："
    dpkg -l | grep -E "^ii +(xserver-xorg-video-nvidia|libnvidia-gl-)"
    echo "!!! 已中止。请先卸掉这些包再继续。"
    exit 1
fi
echo "    OK — 未安装 Xorg nvidia 驱动 / GL 库，显示栈未受影响"

echo
echo "==> [3/4] 设为开机不自动加载（blacklist 只挡自动加载，手动 modprobe 仍可用）"
cat > /etc/modprobe.d/nvidia-manual-load.conf <<'INNER_EOF'
# 开机不自动加载 nvidia 模块，确保启动过程完全不受影响。
# 需要跑 CUDA 时手动执行： sudo modprobe nvidia nvidia_uvm
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
INNER_EOF
update-initramfs -u

echo
echo "==> [4/4] 检查 DKMS 模块构建与签名状态"
dkms status
MODPATH="$(modinfo -n nvidia 2>/dev/null || true)"
if [ -z "$MODPATH" ]; then
    echo "!!! 模块没找到，DKMS 构建可能失败了，检查上面的 dkms status 输出"
    exit 1
fi
echo "    模块路径: $MODPATH"
modinfo nvidia | grep -E "^signer|^sig_key" || echo "!!! 模块未签名 — Secure Boot 下会拒绝加载"

echo
echo "======================================================"
echo "安装完成。现在测试（不需要重启）："
echo "  sudo rmmod nouveau          # 释放显卡（nouveau 当前占着但没在用）"
echo "  sudo modprobe nvidia nvidia_uvm"
echo "  nvidia-smi"
echo "======================================================"
