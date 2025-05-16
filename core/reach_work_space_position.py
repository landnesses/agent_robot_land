#!/usr/bin/env python3
import rospy
import numpy as np
import csv
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from auto_moveit_control import ArmMover

def create_marker_array(points, frame_id="base_link"):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.02
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0  # 蓝色
    marker.color.a = 1.0
    marker.id = 0
    marker.points = points
    marker_array.markers.append(marker)
    return marker_array

def is_position_reachable(mover, x, y, z):
    """
    只进行规划，不执行。返回 True 表示该点可达。
    """
    mover.group.set_start_state_to_current_state()
    mover.group.set_position_target([x, y, z])
    plan_result = mover.group.plan()

    if isinstance(plan_result, tuple):
        try:
            return bool(plan_result[0])
        except Exception:
            return False
    else:
        return bool(plan_result)

if __name__ == "__main__":
    rospy.init_node("reachability_map_generator")
    mover = ArmMover("left_arm")
    pub = rospy.Publisher("/reachable_points", MarkerArray, queue_size=10)

    x_range = np.arange(-0.75, 0.21, 0.025)
    y_range = np.arange(-0.26, 0.73, 0.025)
    z_range = np.arange(0.8, 1.96, 0.025)

    all_points = []
    total_count = len(x_range) * len(y_range) * len(z_range)
    tested_count = 0
    saved_points = []

    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.02
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0  # 蓝色
    marker.color.a = 1.0
    marker.id = 0

    marker_array = MarkerArray()
    marker_array.markers.append(marker)

    for x in x_range:
        for y in y_range:
            for z in z_range:
                tested_count += 1
                progress = (tested_count / total_count) * 100
                print(f"[{tested_count}/{total_count} | {progress:.1f}%] 测试 ({x:.2f}, {y:.2f}, {z:.2f})...")

                if is_position_reachable(mover, x, y, z):
                    pt = Point(x=x, y=y, z=z)
                    marker.points.append(pt)
                    saved_points.append([x, y, z])
                    pub.publish(marker_array)
                    rospy.sleep(0.001)  # 减小等待时间进一步加速

    print(f"[总共可达点数]: {len(marker.points)}，已保存为 CSV 文件")

    with open("reachable_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(saved_points)
