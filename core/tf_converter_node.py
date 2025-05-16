#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

class AppleTFConverter:
    def __init__(self):
        rospy.init_node("apple_tf_converter_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber("/apple_camera_point", PointStamped, self.callback)
        self.marker_pub = rospy.Publisher("/apple_marker", Marker, queue_size=20)

        self.frame_counter = 0
        self.apple_counter = 0
        self.last_frame_time = rospy.Time.now()

    def callback(self, msg):
        try:
            base_point = self.tf_buffer.transform(msg, "base_link", rospy.Duration(1.0))

            current_time = rospy.Time.now()
            # 每轮（每帧）检测前清除上一次的 Marker（以生命周期方式）
            if (current_time - self.last_frame_time).to_sec() > 0.5:
                self.apple_counter = 0
                self.last_frame_time = current_time

            # 创建红球 Marker
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
            self.apple_counter += 1

            rospy.loginfo(f"[Apple {self.apple_counter}] base_link: ({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f}, {marker.pose.position.z:.2f})")

        except Exception as e:
            rospy.logwarn(f"[TF Error] {e}")

if __name__ == "__main__":
    try:
        AppleTFConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
