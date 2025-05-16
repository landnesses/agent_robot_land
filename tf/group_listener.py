#!/usr/bin/env python3
import rospy
import json
import os
from moveit_msgs.msg import MoveGroupActionResult

class MoveGroupResultLogger:
    def __init__(self):
        rospy.init_node("move_group_result_logger", anonymous=True)
        rospy.Subscriber("/move_group/result", MoveGroupActionResult, self.callback)

        self.save_dir = rospy.get_param("~save_dir", "./move_group_logs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.count = 0

        rospy.loginfo("📡 正在监听 /move_group/result ...")

    def callback(self, msg):
        traj = msg.result.planned_trajectory.joint_trajectory
        joint_names = traj.joint_names

        log_data = {
            "header": {
                "seq": msg.header.seq,
                "planning_time": msg.result.planning_time,
                "joint_names": joint_names,
                "point_count": len(traj.points)
            },
            "trajectory": []
        }

        for pt in traj.points:
            log_data["trajectory"].append({
                "time_from_start": pt.time_from_start.to_sec(),
                "positions": list(map(float, pt.positions)),
                "velocities": list(map(float, pt.velocities)) if pt.velocities else [],
                "accelerations": list(map(float, pt.accelerations)) if pt.accelerations else [],
            })

        # 自动保存为 JSON 文件
        filename = f"trajectory_{self.count:03d}.json"
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(log_data, f, indent=2)

        rospy.loginfo(f"✅ 轨迹数据已保存: {path}")
        self.count += 1


if __name__ == "__main__":
    try:
        MoveGroupResultLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
