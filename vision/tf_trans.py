import numpy as np
from scipy.spatial.transform import Rotation as R

class CameraToBaseTransformer:
    def __init__(self, translation, quaternion):
        """
        translation: [xr, yr, zr] 相机在 base_link下的位置
        quaternion: [qx, qy, qz, qw] 相机在 base_link下的方向
        """
        self.translation = np.array(translation)
        self.rotation = R.from_quat(quaternion)  # 四元数转旋转矩阵

    def transform(self, point_in_camera):
        """
        输入物体在相机坐标系下的位置，输出物体在base_link坐标系下的位置
        :param point_in_camera: [x, y, z] in realsense_link
        :return: [x, y, z] in base_link
        """
        point_cam = np.array(point_in_camera)
        point_base = self.rotation.apply(point_cam) + self.translation
        return point_base.tolist()

