#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Bool

class CommandPublisher:
    def __init__(self):
        rospy.init_node('command_publisher_node', anonymous=True)
        self.pub = rospy.Publisher('/detect_command', Bool, queue_size=1)
        self.rate = rospy.Rate(1)  # 1Hz 发布频率

    def run(self):
        while not rospy.is_shutdown():
            msg = Bool(data=True)
            self.pub.publish(msg)
            rospy.loginfo("[command_publisher_node] Published: True")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = CommandPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
