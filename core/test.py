#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState

def publish_arm_command():
    pub = rospy.Publisher('/arm_control', JointState, queue_size=1)
    rospy.init_node('test_arm_control', anonymous=True)
    rospy.sleep(1.0)

    joint_names = [
        'left_fb_joint', 'left_side_joint', 'left_big_arm_joint', 'left_elbow_joint',
        'left_small_arm_joint', 'left_wrist_joint',
        'right_fb_joint', 'right_side_joint', 'right_big_arm_joint', 'right_elbow_joint',
        'right_small_arm_joint', 'right_wrist_joint'
    ]

    # 设置目标角度：左肘 1.0，其它为 0.0
    target_positions = [0.0] * 12
    target_positions[3] = -1.0   # left_elbow_joint

    msg = JointState()
    msg.name = joint_names
    msg.position = target_positions

    pub.publish(msg)
    rospy.loginfo("Sent command to /arm_control")

if __name__ == "__main__":
    publish_arm_command()
