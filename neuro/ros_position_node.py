#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import joblib
import numpy as np
import moveit_commander
from math import radians
from std_msgs.msg import Float32MultiArray


# ==========================  1. 神经网络定义  =========================== #
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


# =======================  2. Position-only refine  ====================== #
def refine_position_only_with_seed(seed_joints, target_xyz,
                                   group_name="left_arm_full",
                                   ee_link=None,
                                   timeout=1.5, attempts=20):
    group = moveit_commander.MoveGroupCommander(group_name)
    robot = moveit_commander.RobotCommander()
    joint_names = group.get_active_joints()

    if ee_link is None:
        ee_link = group.get_end_effector_link()

    group.clear_pose_targets()
    group.set_position_target(target_xyz, end_effector_link=ee_link)

    start_state = robot.get_current_state()
    pos_list = list(start_state.joint_state.position)
    for n, v in zip(joint_names, seed_joints):
        idx = start_state.joint_state.name.index(n)
        pos_list[idx] = float(v)
    start_state.joint_state.position = pos_list
    group.set_start_state(start_state)

    group.set_planning_time(timeout)
    group.set_num_planning_attempts(attempts)

    ret = group.plan()
    if isinstance(ret, tuple):
        success, plan = ret[0], ret[1]
    else:
        plan     = ret
        success  = bool(plan and plan.joint_trajectory.points)

    if not success or not plan.joint_trajectory.points:
        rospy.logwarn("❌ IK / 规划失败，退回 seed_joints")
        return list(seed_joints)

    refined = list(plan.joint_trajectory.points[-1].positions)
    rospy.loginfo("✅ refine 规划成功")
    return refined


# =======================  3. 执行轨迹封装  ====================== #
# =======================  3. 执行轨迹封装（go 版） ====================== #
def execute_joint_plan(joint_values, group_name="left_arm_full"):
    group  = moveit_commander.MoveGroupCommander(group_name)
    robot  = moveit_commander.RobotCommander()
    names  = group.get_active_joints()

    m = radians(0)
    for i, n in enumerate(names):
        joint = robot.get_joint(n)
        lo, hi = joint.bounds()
        raw_val = float(joint_values[i])
        joint_values[i] = float(np.clip(raw_val, lo + m, hi - m))
        if abs(raw_val - joint_values[i]) > 1e-6:
            print(f"⚠️ joint '{i}' 裁剪: {raw_val:.4f} → {joint_values[i]:.4f} (范围: {lo:.4f} ~ {hi:.4f})")

    group.set_start_state_to_current_state()
    group.set_joint_value_target(dict(zip(names, joint_values)))

    rospy.loginfo("🚀 正在执行轨迹 (via go)")
    success = group.go(wait=True)
    group.stop()

    if success:
        rospy.loginfo("✅ 轨迹执行完成")
    else:
        rospy.logerr("❌ 轨迹执行失败")

    return success


# =======================  4. ROS 回调封装  ====================== #
class IKNetRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"✅ 当前推理设备: {self.device}")

        self.model = IKNet().to(self.device)
        self.model.load_state_dict(torch.load("iknet_model_weights.pt", map_location=self.device))
        self.model.eval()
        self.scaler_y = joblib.load("iknet_scaler_y.pkl")

        self.sub = rospy.Subscriber("/iknet_position_input", Float32MultiArray, self.callback)

    def callback(self, msg):
        if len(msg.data) != 4:
            rospy.logwarn("❌ 输入格式必须为 [arm_flag, x, y, z]")
            return
        try:
            arm_flag = int(msg.data[0])
            target_xyz = msg.data[1:]
            group_name = "left_arm_full" if arm_flag == 0 else "right_arm_full"

            nn_input = torch.tensor([msg.data], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                nn_output = self.model(nn_input).cpu().numpy()
            joint_seed = self.scaler_y.inverse_transform(nn_output)[0]
            rospy.loginfo(f" NN 关节角(rad): {np.round(joint_seed, 4).tolist()}")

            refined = refine_position_only_with_seed(joint_seed, target_xyz, group_name)
            rospy.loginfo(f"refine 后关节角(rad): {np.round(refined, 4).tolist()}")

            execute_joint_plan(refined, group_name)
        except Exception as e:
            rospy.logerr(f"❌ 回调处理失败: {e}")


# =======================  5. 主函数入口  ====================== #
if __name__ == '__main__':
    moveit_commander.roscpp_initialize([])
    rospy.init_node("iknet_position_only_refine_node", anonymous=True)
    rospy.loginfo(" 正在监听话题: /iknet_position_input")
    IKNetRunner()
    rospy.spin()
