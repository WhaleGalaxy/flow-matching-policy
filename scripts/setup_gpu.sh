#!/usr/bin/env bash
# 在 Optimus 混合显卡笔记本上，让 NVIDIA 只做 CUDA 计算 + 离屏 Vulkan 渲染，
# 完全不参与显示输出。显示继续由 Intel 核显驱动。
#
# 为什么需要这个脚本、以及为什么早先的做法会导致开机黑屏，见 docs/gpu-setup.md。
# 一句话：nvidia_drm 会抢占 /dev/dri/card0，但它在这类笔记本上没有任何 CRTC，
# Xorg 自动探测选中它之后所有 Screen 都会被删掉，登录界面永远起不来。
#
# 用法： sudo bash scripts/setup_gpu.sh [驱动大版本，默认 570]
set -euo pipefail
[ "$EUID" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }
VER="${1:-570}"

BK="/root/nvidia-rollback-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BK"; cp -a /etc/modprobe.d "$BK/modprobe.d.bak"
dpkg -l | awk '/^ii/{print $2}' > "$BK/pkglist-before.txt"
echo "==> [1/5] 已备份到 $BK"

echo "==> [2/5] 装包之前先堵死 nvidia_drm / nvidia_modeset"
rm -f /etc/modprobe.d/nvidia-manual-load.conf   # 旧方案，纯 blacklist 挡不住，见下
cat > /etc/modprobe.d/nvidia-nodrm.conf <<'EOF'
# N 卡只做计算与离屏渲染，不参与显示。删掉这个文件会导致开机进不了桌面。
#
# 笔记本屏幕接在 Intel 核显上。nvidia_drm 一旦加载会注册成 DRM minor 0、
# 抢走 /dev/dri/card0，而它在这台机器上没有任何显示输出：
#   nvidia 0000:01:00.0: [drm] Cannot find any crtc or sizes
# Xorg 自动探测于是把这块没有输出的卡当成主屏，最终两个 Screen 全被删除：
#   (EE) Screen 0 deleted because of no matching config section.
#   (EE) Screen 1 deleted because of no matching config section.
# → X 起不来，GDM 死循环重试，表现为"开机卡死"。
#
# 关键：单纯 blacklist 挡不住 NVIDIA 自带 udev 规则里的显式 modprobe，
# 必须配合 install ... /bin/false。改完必须 update-initramfs。
blacklist nvidia_drm
blacklist nvidia_modeset
install nvidia_drm /bin/false
install nvidia_modeset /bin/false
EOF

echo "==> [3/5] 安装计算驱动 + Vulkan 用户态库（SAPIEN/ManiSkill 需要 nvidia_icd.json）"
apt-get update
apt-get install -y "nvidia-headless-${VER}" "nvidia-utils-${VER}" "libnvidia-gl-${VER}"

echo "==> [4/5] 重建 initramfs，让屏蔽在早期启动就生效"
update-initramfs -u -k all

echo "==> [5/5] 不重启直接加载并验证"
modprobe -r nouveau 2>/dev/null || true
modprobe nvidia && modprobe nvidia_uvm

fail=0
echo "--- 模块 ---"; lsmod | grep -E "^nvidia" || true
if lsmod | grep -q "^nvidia_drm"; then echo "!! nvidia_drm 竟然加载了，屏蔽没生效"; fail=1; fi
echo "--- 签名（Secure Boot 开启时必须有）---"; modinfo -F signer nvidia || true
echo "--- Vulkan ICD ---"; ls -1 /usr/share/vulkan/icd.d/
echo "--- nvidia-smi ---"; nvidia-smi || fail=1
[ "$fail" -eq 0 ] && echo "OK：CUDA 与离屏 Vulkan 可用，显示栈未受影响" || { echo "有检查未通过，见上"; exit 1; }
