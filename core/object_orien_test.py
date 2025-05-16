import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# 固定左手elbow位置
elbow = np.array([0.31, 0.237, 1.226])
target = np.array([0.31, 0.237, 0.9])  # 初始目标位置

def compute_grasp_axes(target, elbow):
    y_axis = -elbow + target
    y_axis /= np.linalg.norm(y_axis)

    x_down = np.array([0, 0, 1])  
    if abs(np.dot(x_down, y_axis)) > 0.95:  
        x_down = np.array([0, -1, 0])
    
    z_axis = np.cross(x_down, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

# === 初始化可视化 ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.set_zlim(0.5, 1.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 绘制初始点
elbow_scatter = ax.scatter(*elbow, c='blue', s=50, label='Elbow')
target_scatter = ax.scatter(*target, c='red', s=50, label='Target')

# 初始姿态箭头
x_axis, y_axis, z_axis = compute_grasp_axes(target, elbow)
q_x = ax.quiver(*target, *x_axis, length=0.1, color='red', label='X - Down')
q_y = ax.quiver(*target, *y_axis, length=0.2, color='green', label='Y - To Elbow')
q_z = ax.quiver(*target, *z_axis, length=0.1, color='blue', label='Z - Left')

# === 添加滑动条 ===
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.15, 0.02, 0.65, 0.02], facecolor=axcolor)
ax_y = plt.axes([0.15, 0.05, 0.65, 0.02], facecolor=axcolor)
ax_z = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=axcolor)
s_x = Slider(ax_x, 'Target X', -0.6, 0.6, valinit=target[0])
s_y = Slider(ax_y, 'Target Y', -0.6, 0.6, valinit=target[1])
s_z = Slider(ax_z, 'Target Z', 0.5, 1.5, valinit=target[2])

def update(val):
    global target, q_x, q_y, q_z
    target[:] = [s_x.val, s_y.val, s_z.val]
    target_scatter._offsets3d = ([target[0]], [target[1]], [target[2]])

    # 重新计算坐标轴
    x_axis, y_axis, z_axis = compute_grasp_axes(target, elbow)

    # 清除旧箭头
    q_x.remove()
    q_y.remove()
    q_z.remove()

    # 绘制新箭头
    q_x = ax.quiver(*target, *x_axis, length=0.1, color='red')
    q_y = ax.quiver(*target, *y_axis, length=0.2, color='green')
    q_z = ax.quiver(*target, *z_axis, length=0.1, color='blue')

    fig.canvas.draw_idle()

s_x.on_changed(update)
s_y.on_changed(update)
s_z.on_changed(update)

plt.legend()
plt.title("Grasp Pose Axes (Aligned with RViz Left Hand)")
plt.tight_layout()
plt.show()
