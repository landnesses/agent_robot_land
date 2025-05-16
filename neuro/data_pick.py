#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
from math import radians, pi

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

# === 配置项 ===
GROUP_NAME = "left_arm_full"
EE_LINK = "left_endeffector_center_link"
TARGET_SAMPLE_COUNT = 1000
CSV_PATH = "left_arm_ik_dataset.csv"
MAX_NOISE_DEG = 10.0


# === 工具函数 ===
def add_quaternion_noise(quat, max_angle_deg=10):
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    dq = R.from_rotvec(angle_rad * axis).as_quat()
    q = R.from_quat(quat)
    q_new = (q * R.from_quat(dq)).as_quat()
    return q_new

def get_joint_limits(group):
    robot = moveit_commander.RobotCommander()
    joint_names = group.get_active_joints()
    lower = []
    upper = []
    for name in joint_names:
        joint = robot.get_joint(name)
        # 若 JointModel 有 bounds() 方法
        if hasattr(joint, "bounds"):
            l, u = joint.bounds()
        elif hasattr(joint, "min_bound") and hasattr(joint, "max_bound"):
            l, u = joint.min_bound(), joint.max_bound()
        else:
            rospy.logwarn(f"⚠️ 关节 {name} 无法获取限位，使用默认 [-π, π]")
            l, u = -pi, pi
        lower.append(l)
        upper.append(u)
    return joint_names, np.array(lower), np.array(upper)

def compute_fk(joint_names, joint_values, ee_link):
    rospy.wait_for_service('/compute_fk')
    try:
        fk_service = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        request = GetPositionFKRequest()
        request.fk_link_names = [ee_link]
        request.header.frame_id = "base_link"  # 根据你的 tf 树适配

        state = RobotState()
        state.joint_state = JointState()
        state.joint_state.name = joint_names
        state.joint_state.position = joint_values
        request.robot_state = state

        response = fk_service(request)
        if response.error_code.val != response.error_code.SUCCESS:
            raise RuntimeError(f"FK 失败，错误码: {response.error_code.val}")

        pose = response.pose_stamped[0].pose
        pos = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        return pos, quat

    except rospy.ServiceException as e:
        raise RuntimeError(f"无法调用 /compute_fk 服务: {e}")

# === 主函数 ===
def main():
    moveit_commander.roscpp_initialize([])
    rospy.init_node("leftarm_data_gen", anonymous=True)
    group = moveit_commander.MoveGroupCommander(GROUP_NAME)
    rospy.wait_for_service('/compute_fk')

    joint_names, lower, upper = get_joint_limits(group)

    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['arm_flag', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + joint_names
        writer.writerow(header)

        success = 0
        attempt = 0
        while success < TARGET_SAMPLE_COUNT:
            attempt += 1
            sample_joints = np.random.uniform(low=lower, high=upper)
            try:
                pos, quat = compute_fk(joint_names, sample_joints, EE_LINK)
                noisy_quat = add_quaternion_noise(quat, MAX_NOISE_DEG)
                row = [0] + pos + list(noisy_quat) + list(sample_joints)
                writer.writerow(row)
                rospy.loginfo(f"[{success+1}/{TARGET_SAMPLE_COUNT}] OK")
                success += 1
            except Exception as e:
                rospy.logwarn(f"❌ Sample {attempt} failed: {e}")
                continue

    rospy.loginfo(f"✅ 数据采集完成，共保存 {success} 条样本 → {CSV_PATH}")

if __name__ == '__main__':
    main()
