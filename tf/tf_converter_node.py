#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from moveit_commander import PlanningSceneInterface, roscpp_initialize
import threading
import time

class AppleTFConverter:
    def __init__(self):
        rospy.init_node("apple_tf_converter_node")
        roscpp_initialize([])  # 初始化 MoveIt

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scene = PlanningSceneInterface(synchronous=True)
        self.scene_thread = threading.Thread(target=self._wait_for_scene)
        self.scene_ready = False
        self.scene_thread.start()

        self.sub = rospy.Subscriber("/apple_camera_point", PointStamped, self.callback)
        self.marker_pub = rospy.Publisher("/apple_marker", Marker, queue_size=20)

        self.frame_counter = 0
        self.apple_counter = 0
        self.last_frame_time = rospy.Time.now()

    def _wait_for_scene(self):
        # 等待 MoveIt 场景准备就绪
        rospy.sleep(2.0)
        self.scene_ready = True

    def callback(self, msg):
        try:
            base_point = self.tf_buffer.transform(msg, "base_link", rospy.Duration(1.0))

            current_time = rospy.Time.now()
            if (current_time - self.last_frame_time).to_sec() > 0.5:
                self.apple_counter = 0
                self.last_frame_time = current_time

            # Marker 可视化
            marker = Marker()
            marker.header = Header(stamp=current_time, frame_id="base_link")
            marker.ns = "apple"
            marker.id = self.apple_counter
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = base_point.point.x
            marker.pose.position.y = base_point.point.y
            marker.pose.position.z = base_point.point.z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(1.0)
            self.marker_pub.publish(marker)

            # 添加为 MoveIt 碰撞物体
            if self.scene_ready:
                name = f"apple_{self.apple_counter}"
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position = marker.pose.position
                pose.pose.orientation.w = 1.0
                size = (0.025, 0.025, 0.025)  # 立方体

                self.scene.add_box(name, pose, size=size)
                rospy.loginfo(f"✅ 添加碰撞箱: {name} at ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}, {pose.pose.position.z:.2f})")

            self.apple_counter += 1

        except Exception as e:
            rospy.logwarn(f"[TF Error] {e}")

if __name__ == "__main__":
    try:
        AppleTFConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
