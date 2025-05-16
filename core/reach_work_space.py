#!/usr/bin/env python3
import rospy
import numpy as np
import csv
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from auto_moveit_control import ArmMover
from scipy.spatial.transform import Rotation as R

def generate_uniform_quaternions(num_samples=8):
    quaternions = []
    for i in range(num_samples):
        z = 2.0 * i / num_samples - 1.0
        phi = i * 3.6  # golden angle
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        rot = R.from_rotvec(np.array([x, y, z]) * np.pi)
        q = rot.as_quat()
        quaternions.append(q)
    return quaternions

def try_plan_pose(mover, x, y, z, qx, qy, qz, qw):
    pose = mover.group.get_current_pose().pose
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw

    mover.group.set_start_state_to_current_state()
    mover.group.set_pose_target(pose)
    plan_result = mover.group.plan()

    if isinstance(plan_result, tuple):
        try:
            return bool(plan_result[0])
        except Exception:
            return False
    else:
        return bool(plan_result)

def get_color_by_score(score, max_score):
    ratio = score / max_score
    color = ColorRGBA()
    color.r = ratio
    color.g = 1.0 - ratio
    color.b = 0.0
    color.a = 1.0
    return color

def create_marker_array(points, colors, frame_id="base_link"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.02
    marker.color.a = 1.0
    marker.id = 0
    marker.points = points
    marker.colors = colors
    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    return marker_array

if __name__ == "__main__":
    rospy.init_node("reachability_map_generator")
    mover = ArmMover("left_arm")
    pub = rospy.Publisher("/reachable_points", MarkerArray, queue_size=10)

    x_range = np.arange(-0.6, -0.4, 0.05)
    y_range = np.arange(0.45, 0.6, 0.05)
    z_range = np.arange(1.3, 1.6, 0.05)
    orientations = generate_uniform_quaternions(num_samples=8)

    vis_points = []
    vis_colors = []
    reachable_data = []
    total_count = len(x_range) * len(y_range) * len(z_range)
    tested_count = 0

    for x in x_range:
        for y in y_range:
            for z in z_range:
                tested_count += 1
                success_count = 0
                for q in orientations:
                    if try_plan_pose(mover, x, y, z, q[0], q[1], q[2], q[3]):
                        success_count += 1

                print(f"[{tested_count}/{total_count}] [x={x:.2f}, y={y:.2f}, z={z:.2f}] → 成功姿态数: {success_count}/{len(orientations)}" +
                      (" ✅" if success_count > 0 else " ❌"))

                if success_count > 0:
                    pt = Point(x=x, y=y, z=z)
                    color = get_color_by_score(success_count, max_score=len(orientations))
                    vis_points.append(pt)
                    vis_colors.append(color)
                    reachable_data.append((x, y, z, success_count))

                    marker_array = create_marker_array(vis_points, vis_colors)
                    pub.publish(marker_array)
                    rospy.sleep(0.001)

    with open("reachable_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "num_successful_orientations"])
        for entry in reachable_data:
            writer.writerow(entry)

    print(f"[总可达点数]: {len(reachable_data)}，文件已保存：reachable_points.csv")
