import pyrealsense2 as rs
import math
import time

"""
俯仰角pitch：绕x轴旋转，重力加速度与"X-Y平面"的夹角，即重力加速度在Z轴上的分量与重力加速度在"X-Y平面"投影的正切
            正角度是前倾，符合深度和彩色相机Z轴的正方向，负角度是后仰

翻滚角roll：绕z轴旋转，重力加速度与"Z-Y平面"的夹角，即重力加速度在X轴上的分量与重力加速度在"Z-Y平面"投影的正切
            正角度是左相机到右相机，符合深度和彩色相机X轴的正方向 。
"""

if __name__ == "__main__":
    # 相机配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
    pipeline.start(config)

    align_to = rs.stream.accel
    align = rs.align(align_to)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        accel = aligned_frames[0].as_motion_frame().get_motion_data()  # 获取加速度的数据
        # gyro = aligned_frames[1].as_motion_frame().get_motion_data()

        # 各个坐标的线加速度 linear_acceleration:
        ax = aligned_frames[0].as_motion_frame().get_motion_data().x
        ay = aligned_frames[0].as_motion_frame().get_motion_data().y
        az = aligned_frames[0].as_motion_frame().get_motion_data().z

        # 重力加速度在X-Y面上的投影
        pitch_xy = math.sqrt(accel.x * accel.x + accel.y * accel.y)
        # 重力加速度在Z轴上的分量与在X-Y面上的投影的正切，即俯仰角
        pitch = math.atan2(-accel.z, pitch_xy) * 57.3  # 57.3 = 180/3.1415
        # 重力加速度在Z-Y面上的投影
        roll_zy = math.sqrt(accel.z * accel.z + accel.y * accel.y)
        # 重力加速度在X轴上的分量与在Z-Y面上的投影的正切，即翻滚角
        roll = math.atan2(-accel.x, roll_zy) * 57.3

        # 打印姿态角信息
        print("roll:%.3f, pitch:%.3f" % (roll, pitch))
        # print("ax:%.3f, ay:%.3f, az:%.3f" % (ax, ay, az))
        time.sleep(1)

