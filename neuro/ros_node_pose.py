#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import numpy as np
import joblib
from math import radians
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from moveit_msgs.msg import Constraints, OrientationConstraint
import moveit_commander

# ===== 网络结构定义 =====
class IKNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ===== 使用 seed 执行带姿态 + 姿态约束的目标 =====
def execute_pose_plan_with_seed(joint_values, target_xyz, group_name):
    group = moveit_commander.MoveGroupCommander(group_name)
    robot = moveit_commander.RobotCommander()
    joint_names = group.get_active_joints()
    ee_link = group.get_end_effector_link()

    rospy.wait_for_service('/compute_fk')
    fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
    req = GetPositionFKRequest()
    req.fk_link_names = [ee_link]
    req.robot_state.joint_state.name = joint_names
    req.robot_state.joint_state.position = [float(j) for j in joint_values]
    req.header.frame_id = "base_link"
    res = fk_srv.call(req)

    if res.error_code.val != res.error_code.SUCCESS:
        rospy.logwarn("❌ FK 失败，跳过执行")
        return False

    pose = res.pose_stamped[0].pose
    pose.position.x, pose.position.y, pose.position.z = target_xyz

    # 设置姿态约束（±40° 容差）
    oc = OrientationConstraint()
    oc.link_name = ee_link
    oc.header.frame_id = "base_link"
    oc.orientation = pose.orientation
    oc.absolute_x_axis_tolerance = radians(40)
    oc.absolute_y_axis_tolerance = radians(40)
    oc.absolute_z_axis_tolerance = radians(40)
    oc.weight = 1.0

    constraint = Constraints()
    constraint.orientation_constraints.append(oc)
    group.set_path_constraints(constraint)

    group.set_start_state_to_current_state()
    group.set_pose_target(pose)
    group.set_planning_time(2.0)

    # ✅ 使用 go(wait=True) 替代 plan+execute
    rospy.loginfo("🚀 执行目标位姿（含姿态约束）")
    success = group.go(wait=True)
    group.stop()
    group.clear_path_constraints()
    group.clear_pose_targets()

    if success:
        rospy.loginfo("✅ 姿态约束执行完成")
    else:
        rospy.logerr("❌ 执行失败")

    return success


# ===== 回调处理 =====
class IKNetROSWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"✅ 推理设备: {self.device}")

        self.model = IKNet().to(self.device)
        self.model.load_state_dict(torch.load("iknet_model_weights.pt", map_location=self.device))
        self.model.eval()

        self.scaler_y = joblib.load("iknet_scaler_y.pkl")

        self.sub = rospy.Subscriber("/iknet_position_input", Float32MultiArray, self.callback)

    def callback(self, msg):
        if len(msg.data) != 4:
            rospy.logwarn("❌ 输入必须是 [arm_flag, x, y, z]")
            return
        try:
            arm_flag = int(msg.data[0])
            target_xyz = list(msg.data[1:])
            group_name = "left_arm_full" if arm_flag == 0 else "right_arm_full"

            input_tensor = torch.tensor([msg.data], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = self.model(input_tensor).cpu().numpy()
                joint_pred = self.scaler_y.inverse_transform(pred)[0]

            rospy.loginfo(f"神经网络预测角度(rad): {np.round(joint_pred, 4).tolist()}")
            success = execute_pose_plan_with_seed(joint_pred, target_xyz, group_name)
            if not success:
                rospy.logwarn("⚠️ 执行失败")
        except Exception as e:
            rospy.logerr(f"❌ 推理过程出错: {e}")

# ===== 主入口 =====
if __name__ == '__main__':
    moveit_commander.roscpp_initialize([])
    rospy.init_node("iknet_position_pose_refine_node", anonymous=True)
    rospy.loginfo("等待目标输入: /iknet_target_input")
    IKNetROSWrapper()
    rospy.spin()
