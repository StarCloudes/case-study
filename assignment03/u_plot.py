import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图模块

# 读取数值解数据
data = pd.read_csv('u_solution_N256.csv')

# 提取唯一坐标轴
x_vals = np.sort(data['x'].unique())
y_vals = np.sort(data['y'].unique())
X, Y = np.meshgrid(x_vals, y_vals)

# 构造 Z 数组：对应于 u(x, y)
Z = data.pivot(index='y', columns='x', values='u').values

# 创建图形窗口并绘制3D表面
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 添加标题与坐标轴标签
ax.set_title('3D Surface Plot of u(x, y) for N=256')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig("u_surface_N256.png", dpi=300)
plt.show()