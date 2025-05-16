#!/usr/bin/env python3
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

def get_precise_z(group_name="base_car"):
    roscpp_initialize([])
    rospy.init_node("precise_z_reader", anonymous=True)

    group = MoveGroupCommander(group_name)
    pose = group.get_current_pose().pose

    z_val = pose.position.z

    print("=== 当前末端执行器 Pose ===")
    print(f"x: {pose.position.x:.10e}")
    print(f"y: {pose.position.y:.10e}")
    print(f"z: {z_val:.16e}")  # 科学计数法 + 高精度
    print()
    print(f"qx: {pose.orientation.x:.10e}")
    print(f"qy: {pose.orientation.y:.10e}")
    print(f"qz: {pose.orientation.z:.10e}")
    print(f"qw: {pose.orientation.w:.10e}")
    print("==========================")

    roscpp_shutdown()

if __name__ == "__main__":
    get_precise_z("base_car")
