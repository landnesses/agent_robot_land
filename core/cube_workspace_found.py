import pandas as pd
import numpy as np

# 加载 CSV
df = pd.read_csv("leftarm_reachable_points.csv")  # 包含 x, y, z
points = df[['x', 'y', 'z']].values

# 滑动窗口大小（宽、高、深）
window_size = np.array([0.15, 0.15, 0.15])  # [dx, dy, dz]
stride = 0.01

# 点云边界
x_min, y_min, z_min = points.min(axis=0)
x_max, y_max, z_max = points.max(axis=0)

# 限制 y > 0 区域
y_min = max(y_min, 0.28)

x_range = np.arange(x_min, x_max - window_size[0], stride)
y_range = np.arange(y_min, y_max - window_size[1], stride)
z_range = np.arange(z_min, z_max - window_size[2], stride)

best_count = 0
best_min_corner = None

for x0 in x_range:
    for y0 in y_range:
        if y0 < 0:
            continue  # 强制 y > 0
        for z0 in z_range:
            x1, y1, z1 = x0 + window_size[0], y0 + window_size[1], z0 + window_size[2]
            mask = (
                (points[:, 0] >= x0) & (points[:, 0] <= x1) &
                (points[:, 1] >= y0) & (points[:, 1] <= y1) &
                (points[:, 2] >= z0) & (points[:, 2] <= z1)
            )
            count = np.count_nonzero(mask)

            if count > best_count:
                best_count = count
                best_min_corner = (x0, y0, z0)

# 输出结果
print(f"✅ 最佳区域起点: {best_min_corner}")
print(f"✅ 区域大小: {window_size.tolist()}")
print(f"✅ 包含点数: {best_count}")
print(f"建议目标发布区域：")
print(f"x∈[{best_min_corner[0]:.2f}, {best_min_corner[0] + window_size[0]:.2f}], "
      f"y∈[{best_min_corner[1]:.2f}, {best_min_corner[1] + window_size[1]:.2f}], "
      f"z∈[{best_min_corner[2]:.2f}, {best_min_corner[2] + window_size[2]:.2f}]")
# 提取最佳区域内的点
x0, y0, z0 = best_min_corner
x1, y1, z1 = x0 + window_size[0], y0 + window_size[1], z0 + window_size[2]

mask = (
    (points[:, 0] >= x0) & (points[:, 0] <= x1) &
    (points[:, 1] >= y0) & (points[:, 1] <= y1) &
    (points[:, 2] >= z0) & (points[:, 2] <= z1)
)

# 筛选出对应行
best_region_points = df[mask]

# 保存到 CSV 文件
best_region_points.to_csv("best_region_points.csv", index=False)
print("✅ 已保存最优区域点云为 best_region_points.csv")
