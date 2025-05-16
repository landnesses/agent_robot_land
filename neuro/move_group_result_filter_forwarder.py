#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监听 /move_group/result，只转发“执行结果”到 /move_group_result_relay
通过 status.text 内容判断是否是执行结果。
"""

import rospy
from moveit_msgs.msg import MoveGroupActionResult

class ResultFilterForwarder:
    def __init__(self):
        rospy.init_node("move_group_result_filter_forwarder")

        self.sub = rospy.Subscriber("/move_group/result", MoveGroupActionResult, self.callback)
        self.pub = rospy.Publisher("/move_group_result_relay", MoveGroupActionResult, queue_size=10, latch=True)

        rospy.loginfo("🟢 正在监听 /move_group/result 并过滤非执行结果")

    def callback(self, msg):
        text = msg.status.text.lower()  # 转小写方便匹配
        if "executed" in text:
            rospy.loginfo("✅ 检测到执行结果（text包含'executed'），转发到 /move_group_result_relay")
            self.pub.publish(msg)
        else:
            rospy.logwarn(f"⏩ 忽略非执行结果（status.text: {msg.status.text}）")

if __name__ == "__main__":
    try:
        ResultFilterForwarder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
