#!/usr/bin/env python3
import rospy
import csv
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

CSV_FILE = "leftarm_reachable_points.csv"

def load_points_from_csv(filename):
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            x, y, z = map(float, row[:3])
            pt = Point(x=x, y=y, z=z)
            points.append(pt)
    return points

def create_marker(points, frame_id="base_link"):
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
    return marker

if __name__ == "__main__":
    rospy.init_node("csv_reachability_viewer")
    pub = rospy.Publisher("/reachable_points_from_csv", MarkerArray, queue_size=1, latch=True)

    points = load_points_from_csv(CSV_FILE)
    marker = create_marker(points)
    marker_array = MarkerArray()
    marker_array.markers.append(marker)

    rospy.sleep(1.0)  # 等待 RViz 初始化
    pub.publish(marker_array)
    rospy.loginfo(f"[CSV 可达点] 发布完成，共 {len(points)} 点")

    rospy.spin()
