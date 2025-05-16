#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
import math

def print_msg(group, msg):
    if not msg.position:
        return
    print(f"\n[{group.upper()}] 新帧（{rospy.get_time():.3f}s）")
    for i, pos in enumerate(msg.position):
        angle_deg = math.degrees(pos)
        print(f"  关节{i:2d}: {angle_deg: .2f}°")

def main():
    rospy.init_node("joint_state_real_time_printer")

    # ✅ 控制组菜单
    groups = {
        "1": ("left", "/left_arm_control_test"),
        "2": ("right", "/right_arm_control_test"),
        "3": ("waist", "/waist_control_test")
    }

    print("请选择要监听的控制组：")
    print("  1. 左臂 (left)")
    print("  2. 右臂 (right)")
    print("  3. 腰部 (waist)")

    choice = input("请输入编号 [1-3]: ").strip()
    if choice not in groups:
        print("❗ 无效输入，退出")
        return

    group, topic = groups[choice]
    print(f"监听中：{topic}（单位：度）...")

    rospy.Subscriber(topic, JointState, lambda msg: print_msg(group, msg))
    rospy.spin()

if __name__ == "__main__":
    main()
