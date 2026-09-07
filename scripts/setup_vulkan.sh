#!/usr/bin/env bash
# 给 ManiSkill3 / SAPIEN 的渲染器提供 NVIDIA Vulkan ICD。
#
# 只装 libnvidia-gl-570（GL/EGL/Vulkan 库），仍然**不装** xserver-xorg-video-nvidia，
# 所以 Xorg 继续用 Intel 核显。libglvnd 的设计就是让 Mesa 和 NVIDIA 的 GL 共存，
# 由 X server 按屏幕决定给客户端哪个 vendor。
#
# 安全性：装完不重启 Xorg，当前会话不受影响；脚本会当场验证显示栈仍走 Mesa。
# 退路： sudo apt purge libnvidia-gl-570 && sudo apt autoremove
#
# 用法： sudo bash setup_vulkan.sh
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }

echo "==> [1/4] 安装 libnvidia-gl-570 + 诊断工具"
apt-get update
apt-get install -y libnvidia-gl-570 vulkan-tools mesa-utils

echo
echo "==> [2/4] 确认没有把 Xorg 显示驱动带进来"
if dpkg -l | grep -qE "^ii +xserver-xorg-video-nvidia"; then
    echo "!!! 危险：装上了 Xorg nvidia 驱动，重启会黑屏。立刻卸载："
    echo "    sudo apt purge xserver-xorg-video-nvidia-570"
    exit 1
fi
echo "    OK — 未安装 Xorg nvidia 驱动"

echo
echo "==> [3/4] 确认显示仍然走 Intel/Mesa（重启前就能验证）"
RENDERER="$(sudo -u "${SUDO_USER:-$(logname)}" DISPLAY=:0 glxinfo 2>/dev/null | grep -i 'OpenGL renderer' || true)"
echo "    ${RENDERER:-<取不到，可能不在图形会话中，可忽略>}"
if echo "$RENDERER" | grep -qi nvidia; then
    echo "!!! 注意：显示栈切到 NVIDIA 了，重启有风险。建议 purge libnvidia-gl-570 退回。"
    exit 1
fi

echo
echo "==> [4/4] 检查 Vulkan ICD 与设备枚举"
ls -1 /usr/share/vulkan/icd.d/
echo
modprobe nvidia nvidia_uvm 2>/dev/null || true
if vulkaninfo --summary 2>/dev/null | grep -A4 -i "GPU0\|deviceName"; then
    :
else
    echo "!!! vulkaninfo 没能枚举到设备。可能是 focal 的 Vulkan loader (1.2.131) 太老，"
    echo "    或 nvidia 模块没加载。先确认： sudo modprobe nvidia nvidia_uvm && nvidia-smi"
fi

echo
echo "======================================================"
echo "完成。把上面 [3/4] 和 [4/4] 的输出发回来。"
echo "关键看两点： renderer 仍是 Intel/Mesa；ICD 列表里出现 nvidia_icd.json"
echo "======================================================"
