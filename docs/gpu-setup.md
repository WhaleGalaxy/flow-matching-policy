# 在 Optimus 笔记本上配 GPU：一次开机黑屏的排查

这份记录和 [debugging.md](debugging.md) 是一类东西——结论很短（屏蔽一个内核模块），
但从"开机进不去系统"定位到根因的过程值得留下来。踩过两次同一个坑之后写的。

## 环境

Lenovo Legion Y9000P IRX8：i9-13900HX 核显 + RTX 4060 Laptop，Optimus 混合显卡，
**笔记本屏幕物理上接在 Intel 核显上**。Ubuntu 20.04.6 + HWE 内核 5.15，Secure Boot 开启。

## 症状

装完 NVIDIA 驱动重启后，开机停在黑屏，进不了登录界面。三次都是同一个表现。
只能 Ctrl+Alt+F3 进 TTY，`apt purge '^nvidia-.*'` 之后才能恢复图形界面。

关键观察：**系统其实没死**。事后 `journalctl -b -1` 显示日志一路跑到我按下电源键为止，
cron 都正常触发了。死的只是显示。

## 根因

内核日志里两行相邻的记录说明了一切：

```
[drm] Initialized nvidia-drm 0.0.0 20160202 for 0000:01:00.0 on minor 0
nvidia 0000:01:00.0: [drm] Cannot find any crtc or sizes
```

`nvidia_drm` 注册成了 **DRM minor 0**，也就是抢走了 `/dev/dri/card0`；
但这块卡在 Optimus 笔记本上**没有接任何显示输出**，所以没有 CRTC。

Xorg 的自动探测按 DRM 设备顺序选主屏，于是选中了这块没有输出的卡：

```
(==) Matched modesetting as autoconfigured driver 1
(EE) [drm] Failed to open DRM device for (null): -2
(II) modeset(G0): using drv /dev/dri/card0
(EE) Screen 0 deleted because of no matching config section.
(EE) Screen 1 deleted because of no matching config section.
```

两个 Screen 全被删掉，X 起不来，GDM 无限重试 —— 表现为"开机卡死"。

注意 `modeset(G0)`：Intel 那块被降级成了 GPUDevice（从属设备），而不是 Screen 0。
主次关系整个反了。

## 两个把排查带偏的岔路

**岔路一：以为是 Secure Boot。** Secure Boot 确实开着，日志里也有
`Kernel is locked down from EFI Secure Boot mode` 和 `Lockdown: Xorg: raw io port access is restricted`。
但这两条都是无害的：MOK 密钥早就注册过，`modinfo -F signer nvidia` 能读到
签名者，而且日志清清楚楚显示模块**加载成功了**。Xorg 的 raw IO 端口访问被限制
对 `modesetting` 驱动没有影响。**模块签名不是问题，不需要关 Secure Boot。**

**岔路二：以为 `blacklist` 就能挡住模块。** 先前的做法是写一份 blacklist：

```
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
```

然后满以为开机不会加载。实际上内核日志显示**模块照样全部加载了**。原因是
`blacklist` 只阻止通过**模块别名**的自动加载，挡不住 NVIDIA 自带 udev 规则里的
**显式 `modprobe`**。而且改完没有重建 initramfs。

## 正确做法

见 [`scripts/setup_gpu.sh`](../scripts/setup_gpu.sh)。三个要点：

1. **只屏蔽 `nvidia_drm` 和 `nvidia_modeset`，保留 `nvidia` 和 `nvidia_uvm`。**
   CUDA 和离屏 Vulkan 渲染都不需要 DRM 模块。屏蔽的是"让 N 卡去点亮显示器"
   这个能力，不是 N 卡本身。
2. **必须用 `install <模块> /bin/false`**，光靠 `blacklist` 挡不住显式 modprobe。
3. **改完 `update-initramfs -u`**，否则早期启动阶段不生效。

装完之后：

| | 状态 |
|---|---|
| `nvidia-smi` | ✅ RTX 4060 Laptop, 570.133.07, CUDA 12.8 |
| `torch.cuda.is_available()` | ✅ |
| SAPIEN / ManiSkill 离屏渲染 | ✅ `nvidia_icd.json` 在 ICD 列表里 |
| 桌面显示 | ✅ 仍走 Intel + Mesa，完全不受影响 |
| `lsmod \| grep nvidia_drm` | 空（这正是我们要的） |

代价只有两样：**外接显示器点不亮**（Legion 的 HDMI 口直连独显），
以及 **ManiSkill 的交互式 viewer 开不了窗口**（离屏渲染、录视频、评测都正常）。
对这个项目来说都不影响。

## 万一又黑屏了

不用进 TTY 删包。GRUB 界面按 `e`，在 `linux` 那行末尾加：

```
modprobe.blacklist=nvidia_drm,nvidia_modeset,nvidia
```

F10 启动即可。
