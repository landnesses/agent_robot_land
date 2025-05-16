#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys

# 添加 yolov5 路径
yolov5_path = os.path.join(os.path.dirname(__file__), 'YOLOV5')
sys.path.append(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class YoloROSViewer:
    def __init__(self):
        rospy.init_node('yolo_ros_live_viewer', anonymous=True)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)

        self.bridge = CvBridge()
        self.rgb_image = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(os.path.join(yolov5_path, 'best.pt'), map_location=self.device)
        self.model.eval()
        self.imgsz = 640  # 输入分辨率

        rospy.loginfo("YOLOv5 Live Viewer initialized. Waiting for images...")
        self.run()

    def image_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("CV bridge error: %s" % e)

    def detect(self, frame):
        img = letterbox(frame, new_shape=self.imgsz)[0]
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
        img_rgb = np.ascontiguousarray(img_rgb)

        img_tensor = torch.from_numpy(img_rgb).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = self.model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)
        results = []

        if pred[0] is not None and len(pred[0]) > 0:
            pred[0][:, :4] = scale_coords(img_tensor.shape[2:], pred[0][:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred[0]:
                results.append((xyxy, float(conf), int(cls)))
        return results

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.rgb_image is not None:
                frame = self.rgb_image.copy()
                detections = self.detect(frame)

                for (x1, y1, x2, y2), conf, cls in detections:
                    label = f"apple {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("YOLOv5 Apple Viewer", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        viewer = YoloROSViewer()
    except rospy.ROSInterruptException:
        pass
