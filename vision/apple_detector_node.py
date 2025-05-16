#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys

# YOLOv5 模型路径
yolov5_path = os.path.join(os.path.dirname(__file__), 'YOLOV5')
sys.path.append(yolov5_path)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

class AppleDetectorNode:
    def __init__(self):
        rospy.init_node('apple_detector_node', anonymous=True)

        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)
        self.command_sub = rospy.Subscriber("/detect_command", Bool, self.command_callback, queue_size=1)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info_received = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights_path = os.path.join(yolov5_path, 'best.pt')
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        self.imgsz = 640

        self.point_pub = rospy.Publisher("/apple_camera_point", PointStamped, queue_size=10)

        rospy.loginfo("Apple Detector Node ready.")

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.camera_info_received = True

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

    def command_callback(self, msg):
        if msg.data:
            rospy.loginfo("Received detect command. Running detection...")
            self.detect_apple()

    def detect_apple(self):
        if self.rgb_image is None or self.depth_image is None or not self.camera_info_received:
            rospy.logwarn("Waiting for image/depth/intrinsics.")
            return

        img = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz))
        img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).to(self.device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        pred = self.model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)

        if pred[0] is None or len(pred[0]) == 0:
            rospy.loginfo("No apples detected.")
            return

        pred = pred[0]
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], self.rgb_image.shape).round()

        for i, det in enumerate(pred):
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            h, w = self.depth_image.shape[:2]
            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)
            z = self.depth_image[v, u] * 0.001 if self.depth_image.dtype == np.uint16 else self.depth_image[v, u]

            if z <= 0:
                continue

            X = -(u - self.cx) * z / self.fx
            Y = -(v - self.cy) * z / self.fy
            Z = z

            point = PointStamped()
            point.header.stamp = rospy.Time.now()
            point.header.frame_id = "realsense_link"
            point.point.x = X
            point.point.y = Y
            point.point.z = Z

            self.point_pub.publish(point)
            rospy.loginfo(f"[Apple {i+1}] (u={u}, v={v}) => (X={X:.3f}, Y={Y:.3f}, Z={Z:.3f})")

if __name__ == "__main__":
    try:
        AppleDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
