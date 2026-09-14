"""架构图：三条臂共用什么、只在哪里分叉。

这张图存在的理由是**上一版画错了内容**。上一版只画了 Flow Matching 一条通路
（噪声 → 速度场 → Euler 积分），Diffusion 和 BC 只出现在图注文字里。可是本项目
的主张不是"FM 长这样"，而是"三条臂只差一个部件，所以成功率的差异只能归因到
那个部件"。主图没画出受控对比，就等于没画出结论。

于是配色本身承载论证：

    灰框 = 三条臂逐层相同（编码器、context、4 层 Transformer 骨干、参数量）
    彩框 = 这条臂独有（喂进骨干的 x、时间条件、输出被解释成什么、损失、采样）

同理，最右边 BC 那列**没有回环虚线**，FM 是 ×10、Diffusion 是 ×100 ——
延迟差距在图上是直接可见的，不需要额外一句话去说。

图里的数字都来自仓库里已有的结果，不在这里重算，改动时对着这三处核：
  - 参数量 6.80M / BC 多 112 个 query 参数      → README「核心思路 · 基线的可比性」
  - 采样步数与延迟 14.3 / 133.6 ms              → outputs/pareto_points.csv
  - 单任务成功率 61±15 / 13±9 / 53±19           → README「单任务受控对比」主结果表

    python scripts/draw_architecture.py                  # docs/figures/architecture.png
    python scripts/draw_architecture.py --dpi 300 --svg  # 讲演用：矢量 + 高分辨率
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.font_manager  # noqa: F401  (填充 fontManager)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

# 中文字体：matplotlib 不会自动挑 CJK 字体，挑不到就把汉字画成一排方框。
# 只把**本机装了的**候选写进 font.family —— 直接列一串候选虽然也能工作，但每画
# 一个 text 都会为缺失的那几个刷一行 findfont 警告。
_CJK_CANDIDATES = ["Noto Sans CJK SC", "Noto Sans CJK JP", "PingFang SC",
                   "Microsoft YaHei", "Source Han Sans SC", "WenQuanYi Zen Hei",
                   "Droid Sans Fallback"]
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [n for n in _CJK_CANDIDATES if n in _installed] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

INK, MUTED = "#1E2733", "#5B6773"
SHARED_FC, SHARED_EC = "#EDF1F4", "#8A97A3"      # 共用 = 中性灰
CTX_EC = "#7052A8"
METH = {
    "fm":   ("#FCEBEA", "#C04A44", "Flow Matching（Ours）"),
    "ddpm": ("#E9F0FB", "#3F6FBF", "Diffusion Policy（DDPM）"),
    "bc":   ("#E9F5EC", "#2F7D5B", "Behavior Cloning（cross-attn）"),
}

# 画布 16in 宽 = 160 数据单位，所以 1pt = 1/72 in = 0.1389 单位。box() 靠这个换算
# 把字号排成行高；写死成别的数就会溢出框（第一版正是这么糊掉的）。
PT = 0.1389
LINE_SPACING = 1.62


def box(ax, x, y, w, h, lines, fc, ec, lw=1.6):
    """lines = [(文本, 字号pt, 'b' 粗体 | 'm' 次要)]，整体在框内垂直居中。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2))
    heights = [size * PT * LINE_SPACING for _, size, _ in lines]
    cy = y + h / 2 + sum(heights) / 2
    for (txt, size, style), lh in zip(lines, heights):
        cy -= lh / 2
        ax.text(x + w / 2, cy, txt, ha="center", va="center", fontsize=size,
                color=INK if style == "b" else MUTED,
                fontweight="bold" if style == "b" else "normal", zorder=3)
        cy -= lh / 2


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.6, rad=0.0, ls="-", head="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=head, mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ── 三条臂的差异，逐格写死；相同的部分由 draw() 统一画成灰框 ──────────────
SPEC = {
    "fm": dict(
        inp=[("噪声 + 干净动作的直线插值", 10, "b"),
             (r"$x_t=(1-t)\,x_0+t\,x_1$", 11.5, "m"),
             ("流时间 t ~ U(0,1)", 9.2, "m")],
        out=[("预测条件速度场  " + r"$v_\theta$", 10.5, "b"),
             (r"MSE $\;\|\,v_\theta(x_t,t,c)-(x_1-x_0)\,\|^2$", 10, "m"),
             ("目标是常数，条件路径是直线", 9.2, "m")],
        smp=[("Euler 积分 · 10 步", 11, "b"),
             (r"$x \leftarrow x+\Delta t\cdot v_\theta$", 11, "m")],
        res=[("16×7 动作块 · 执行前 8 步", 10.5, "b"),
             ("14.3 ms   ·   单任务 61 ± 15%", 10, "m")],
        loop="× 10 次"),
    "ddpm": dict(
        inp=[("按 cosine 调度前向加噪", 10, "b"),
             (r"$x_k=\sqrt{\bar{\alpha}_k}\,x_1+\sqrt{1-\bar{\alpha}_k}\,\epsilon$", 11.5, "m"),
             ("噪声步 k ~ U{0…99}", 9.2, "m")],
        out=[("预测所加的噪声  " + r"$\epsilon_\theta$", 10.5, "b"),
             (r"MSE $\;\|\,\epsilon_\theta(x_k,k,c)-\epsilon\,\|^2$", 10, "m"),
             ("与 FM 的 loss 不可比（目标不同）", 9.2, "m")],
        smp=[("DDIM 逐步去噪 · 100 步", 11, "b"),
             (r"$x_0$ 预测须裁剪到 ±4，否则基线归零", 9.2, "m")],
        res=[("16×7 动作块 · 执行前 8 步", 10.5, "b"),
             ("133.6 ms   ·   单任务 13 ± 9%", 10, "m")],
        loop="× 100 次"),
    "bc": dict(
        inp=[("可学习的常量 query（16×7）", 10, "b"),
             ("无噪声输入 · 无流时间", 11, "m"),
             ("(AdaLN 需要 t，恒喂 1.0)", 9.2, "m")],
        out=[("直接输出动作块", 10.5, "b"),
             (r"MSE $\;\|\,a_\theta(c)-x_1\,\|^2$", 10, "m"),
             ("学到的是条件均值 E[a|obs]", 9.2, "m")],
        smp=[("一次前向 · 没有采样循环", 11, "b"),
             ("推理最快，但只给一个确定值", 9.2, "m")],
        res=[("16×7 动作块 · 执行前 8 步", 10.5, "b"),
             ("1 次前向   ·   单任务 53 ± 19%", 10, "m")],
        loop=None),
}

ENCODERS = [
    (2,   "RGB 观测", "2 帧 · 126×126", "DINOv2-S (ViT-S/14)", "冻结 → 81 个 patch token"),
    (55,  "语言指令", '"pick up the cube"', "SigLIP 文本塔", "冻结 → 1 个 token"),
    (108, "本体状态 12 维", "关节角+夹爪 9 + 目标位置 3", "Proprio MLP", "可训练 → 1 个 token"),
]
COLS = [(6, "fm"), (58, "ddpm"), (110, "bc")]
W = 46
Y_HEAD, Y_INP, Y_BB, Y_OUT, Y_SMP, Y_RES = 57.0, 46.5, 34.8, 23.0, 12.6, 3.4
H_INP, H_BB, H_OUT, H_SMP, H_RES = 9.0, 10.2, 10.2, 8.8, 8.0


def draw():
    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 97)
    ax.axis("off")

    # ── ① 条件通路：三条臂完全共用，所以全部画成灰框 ──────────────────
    ax.text(2, 94.4, "①  条件通路 —— 三条臂完全共用", fontsize=13, color=INK,
            fontweight="bold")
    ax.text(60, 94.6, "视觉 / 语言骨干全部冻结（占 99.5% 参数），演示只有 1000 条/任务",
            fontsize=10, color=MUTED)

    for gx, t1, s1, t2, s2 in ENCODERS:
        box(ax, gx, 81, 22, 8.6, [(t1, 11, "b"), (s1, 8.8, "m")], "#FFFFFF", SHARED_EC, lw=1.4)
        arrow(ax, gx + 22, 85.3, gx + 26, 85.3)
        box(ax, gx + 26, 81, 24, 8.6, [(t2, 11, "b"), (s2, 8.8, "m")], SHARED_FC, SHARED_EC)
        arrow(ax, gx + 38, 81, gx + 38, 77.6, color=CTX_EC)

    box(ax, 2, 69.5, 156, 8.1,
        [("Context：83 个 token × 256 维     =     81 patch  +  1 语言  +  1 本体/目标",
          12.5, "b")], "#F1EDF9", CTX_EC, lw=1.8)

    # ── ② 动作头：唯一被替换的部件 ──────────────────────────────────
    ax.text(2, 65.0, "②  动作头 —— 唯一被替换的部件（三选一）", fontsize=13, color=INK,
            fontweight="bold")
    ax.text(66, 65.2, "灰色框 = 三条臂逐层相同；彩色框 = 这条臂独有", fontsize=10, color=MUTED)

    for cx, key in COLS:
        fc, ec, name = METH[key]
        spec = SPEC[key]

        box(ax, cx, Y_HEAD, W, 5.6, [(name, 12.5, "b")], fc, ec, lw=1.9)
        arrow(ax, cx + W / 2, Y_HEAD, cx + W / 2, Y_INP + H_INP, color=ec)

        box(ax, cx, Y_INP, W, H_INP, spec["inp"], fc, ec)
        arrow(ax, cx + W / 2, Y_INP, cx + W / 2, Y_BB + H_BB, color=ec)

        box(ax, cx, Y_BB, W, H_BB,
            [("共享骨干：4 层 Transformer · 6.80M", 10.5, "b"),
             ("self-attn  +  cross-attn(← context)  +  AdaLN", 9.2, "m"),
             ("三条臂逐层同构，参数量完全相同", 9.2, "m")],
            SHARED_FC, SHARED_EC, lw=1.9)
        arrow(ax, cx + W / 2, Y_BB, cx + W / 2, Y_OUT + H_OUT, color=ec)

        box(ax, cx, Y_OUT, W, H_OUT, spec["out"], fc, ec)
        arrow(ax, cx + W / 2, Y_OUT, cx + W / 2, Y_SMP + H_SMP, color=ec)

        box(ax, cx, Y_SMP, W, H_SMP, spec["smp"], fc, ec)
        arrow(ax, cx + W / 2, Y_SMP, cx + W / 2, Y_RES + H_RES, color=ec)

        box(ax, cx, Y_RES, W, H_RES, spec["res"], "#FFFFFF", ec, lw=1.4)

        # context 经 cross-attention 进骨干（走列外侧的虚线，不穿过任何框）
        spine = cx - 3.6
        arrow(ax, spine, 69.5, spine, Y_BB + H_BB / 2, color=CTX_EC, lw=1.3,
              ls=(0, (3, 2.6)), head="-")
        arrow(ax, spine, Y_BB + H_BB / 2, cx, Y_BB + H_BB / 2, color=CTX_EC, lw=1.3)

        # 采样循环。BC 没有这条弧，这个"缺口"就是延迟差距的来源
        if spec["loop"]:
            y_from, y_to = Y_SMP + H_SMP / 2, Y_INP + 0.5
            arrow(ax, cx + W, y_from, cx + W, y_to, color=ec, lw=1.5, rad=-0.16,
                  ls=(0, (4, 3)))
            ax.text(cx + W + 4.2, (y_from + y_to) / 2, spec["loop"], ha="center",
                    va="center", fontsize=9.5, color=ec, fontweight="bold", rotation=90,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none"))

    ax.text(2, 0.4,
            "唯一的变量是动作头：三条臂共用同一编码器、同一骨干、同一 6.80M 参数量"
            "（BC 只多一个 query 的 112 个）、同一训练预算与观测配置。"
            "另有反面对照 BC·平均池化：把 83 个 token 平均后过 MLP，目标被稀释成 1/83，单任务 3%。",
            fontsize=9.6, color=MUTED)

    fig.tight_layout(pad=0.4)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/figures/architecture.png"))
    ap.add_argument("--dpi", type=int, default=150,
                    help="README 用 150 就够清晰；讲演/海报用 300")
    ap.add_argument("--svg", action="store_true", help="同时写一份同名 .svg（矢量，插 PPT 不糊）")
    args = ap.parse_args()

    fig = draw()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")
    if args.svg:
        svg = out.with_suffix(".svg")
        fig.savefig(svg, facecolor="white", bbox_inches="tight")
        print(f"wrote {svg}")


if __name__ == "__main__":
    main()
