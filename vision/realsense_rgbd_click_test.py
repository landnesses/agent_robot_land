#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RealSenseRGBDViewer:
    def __init__(self):
        rospy.init_node('realsense_rgbd_click_test', anonymous=True)

        # 订阅 RGB 图像
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        # 订阅 深度图像（对齐到RGB）
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        # 创建窗口，绑定鼠标点击事件
        cv2.namedWindow("RGB Image")
        cv2.setMouseCallback("RGB Image", self.mouse_callback)

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("RGB conversion failed: %s" % str(e))

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Depth conversion failed: %s" % str(e))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_image is not None:
                # 注意深度图是float或者uint16，看实际情况
                depth_value = self.depth_image[y, x]

                # 有些深度图单位是毫米（uint16），要换算成米
                if self.depth_image.dtype == np.uint16:
                    depth_in_meters = depth_value * 0.001  # 毫米 -> 米
                else:
                    depth_in_meters = depth_value  # 已经是米

                rospy.loginfo(f"Clicked at (u={x}, v={y}), depth={depth_in_meters:.3f} m")
                # 画个小圆圈标记
                if self.rgb_image is not None:
                    cv2.circle(self.rgb_image, (x, y), 5, (0, 0, 255), -1)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.rgb_image is not None:
                cv2.imshow("RGB Image", self.rgb_image)
            if self.depth_image is not None:
                depth_vis = self.normalize_depth(self.depth_image)
                cv2.imshow("Depth Image", depth_vis)

            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
            rate.sleep()

        cv2.destroyAllWindows()

    def normalize_depth(self, depth):
        """把深度图归一化到可视化范围（伪彩色显示）"""
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001  # 毫米 -> 米

        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = np.uint8(depth_norm)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return depth_vis

if __name__ == '__main__':
    viewer = RealSenseRGBDViewer()
    viewer.run()
