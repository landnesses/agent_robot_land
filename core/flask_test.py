#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)

class MarkerServer:
    def __init__(self):
        rospy.init_node("marker_server_node", anonymous=True)
        self.last_stamp = None

        self.lock = threading.Lock()
        self.current_markers = []
        self.last_reset_time = time.time()

        self.sub = rospy.Subscriber("/apple_marker", Marker, self.marker_callback)
        threading.Thread(target=self.run_flask, daemon=True).start()

        rospy.loginfo("✅ MarkerServer ready at http://localhost:5008/get_latest_detections")

    def marker_callback(self, msg: Marker):
        with self.lock:
            # 判断是否为新一轮（基于 ROS 时间）
            if self.last_stamp is None or (msg.header.stamp - self.last_stamp).to_sec() > 0.1:
                self.current_markers.clear()
                self.last_stamp = msg.header.stamp
            self.current_markers.append(msg)


    def run_flask(self):
        app.run(host="0.0.0.0", port=5008)

server = MarkerServer()

@app.route("/get_latest_detections", methods=["GET"])
def get_latest():
    with server.lock:
        if not server.current_markers:
            return jsonify({"success": False, "message": "No apples detected."}), 404

        result = []
        for m in server.current_markers:
            result.append({
                "id": m.id,
                "x": m.pose.position.x,
                "y": m.pose.position.y,
                "z": m.pose.position.z,
                "frame": m.header.frame_id,
                "stamp": m.header.stamp.to_sec()
            })

        return jsonify({
            "success": True,
            "count": len(result),
            "apples": result
        })

if __name__ == "__main__":
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
