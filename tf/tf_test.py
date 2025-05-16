from scipy.spatial.transform import Rotation as R
import numpy as np

# Provided elbow and target
elbow = np.array([-0.31, 0.237, 1.226])
target = np.array([-0.09, 0.79, 1.44])

# Compute grasp axes
def compute_grasp_axes(target, elbow):
    y_axis = elbow - target
    y_axis /= np.linalg.norm(y_axis)

    x_down = np.array([0, 0, -1])
    if abs(np.dot(x_down, y_axis)) > 0.95:
        x_down = np.array([0, -1, 0])
    
    z_axis = np.cross(x_down, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

x_axis, y_axis, z_axis = compute_grasp_axes(target, elbow)

# Construct rotation matrix and convert to quaternion
R_mat = np.column_stack((x_axis, y_axis, z_axis))
rotation = R.from_matrix(R_mat)
quat = rotation.as_quat()  # Returns in [x, y, z, w] order

print(quat)
