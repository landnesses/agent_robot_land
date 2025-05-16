#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_frames = np.asanyarray(aligned_depth_frame.get_data())
        color_frame = frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        
        cv2.namedWindow('realsense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realsense', color_frame)
        cv2.imshow('dddd',aligned_frames)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()



print("ssssssssssss")


# def callback(data):
#     rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

# def listener():

#     rospy.init_node('listener', anonymous=True)
 
#     rospy.Subscriber("chatter", String, callback)
 
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()
 
# if __name__ == '__main__':
#     listener()
