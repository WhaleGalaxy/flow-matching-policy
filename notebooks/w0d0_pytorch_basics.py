"""Week0 Day0 热身：手搓线性回归，只用 tensor + backward()，不碰 nn.Module。

目的不是"学会线性回归"，而是确认你对这三件事的心智模型是对的：
  1. requires_grad=True 的 tensor 才会被 autograd 追踪
  2. loss.backward() 把梯度累加到 .grad 上（是累加，不是覆盖 —— 所以要 zero_()）
  3. 更新参数要用 .data 或 no_grad()，否则更新本身会被记进计算图
"""
import torch

torch.manual_seed(0)

# 造数据： y = 2x + 1 + 噪声
x = torch.randn(100, 1)
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.05
for step in range(500):
    loss = ((x @ w + b - y) ** 2).mean()
    loss.backward()

    # 为什么用 .data？直接 w -= ... 会被 autograd 追踪成一次运算，
    # 而且 w 是叶子节点、原地改会报错。等价写法是 with torch.no_grad():
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    # 关键：梯度是累加的，不清零的话下一步会把这一步的梯度也算进去
    w.grad.zero_()
    b.grad.zero_()

    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

print(f"\n最终 w={w.item():.4f} (真值 2.0), b={b.item():.4f} (真值 1.0)")
assert abs(w.item() - 2.0) < 0.05 and abs(b.item() - 1.0) < 0.05, "没收敛到真值"
print("OK — autograd 行为符合预期")
