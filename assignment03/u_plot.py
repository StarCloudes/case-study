import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# Load residual data
data = pd.read_csv('u_solution_N256.csv')
# Extract x, y, and z values
x_vals = np.sort(data['x'].unique())
y_vals = np.sort(data['y'].unique())
X, Y = np.meshgrid(x_vals, y_vals)
Z = data.pivot(index='y', columns='x', values='u').values
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Surface Plot of u(x, y) for N=256')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig("u_surface_N256.png", dpi=300)
plt.show()