#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

# import argparse
# import os
# import platform
# import sys
# from pathlib import Path

# import cv2
# import numpy as np
# import tqqorch
# import torch.backends.cudnn as cudnn

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)         #定义话题名chatter
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()    #rate.sleep(int)错误写法
        rospy.sleep(1)

if __name__=="__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass