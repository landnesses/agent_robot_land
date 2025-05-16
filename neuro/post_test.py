# iknet_infer_position_only.py：位置 → 推理 → refine → 加入姿态 + 姿态约束 → 执行

import torch
import torch.nn as nn
import numpy as np
import joblib
import readline
import rospy
import moveit_commander
from math import radians
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, OrientationConstraint

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

    group.set_pose_target(pose)
    group.set_start_state_to_current_state()
    group.set_planning_time(2.0)

    ok, plan, _, _ = group.plan()

    group.clear_path_constraints()

    if not ok:
        rospy.logerr("❌ 轨迹规划失败")
        return False

    group.execute(plan, wait=True)
    group.stop()
    rospy.loginfo("✅ 姿态约束执行完成")
    return True

# ===== 主推理流程 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 当前推理设备:", device)

    model = IKNet().to(device)
    model.load_state_dict(torch.load("iknet_model_weights.pt", map_location=device))
    model.eval()
    scaler_y = joblib.load("iknet_scaler_y.pkl")

    moveit_commander.roscpp_initialize([])
    rospy.init_node("iknet_position_pose_refine", anonymous=True)

    print("🧪 输入: arm_flag x y z，空格分隔。arm_flag 为 0 或 1")

    while not rospy.is_shutdown():
        line = input("➤ 输入目标位置: ").strip()
        if line.lower() in {"exit", "quit"}:
            print("👋 退出")
            break
        try:
            tokens = list(map(float, line.split()))
            if len(tokens) != 4:
                print("❌ 必须输入 4 个数值：arm_flag x y z")
                continue
            arm_flag = int(tokens[0])
            target_xyz = tokens[1:]
            group_name = "left_arm_full" if arm_flag == 0 else "right_arm_full"

            input_tensor = torch.tensor([tokens], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy()
                joint_pred = scaler_y.inverse_transform(pred)[0]

            print("✅ 推理关节角度 (rad):")
            for i, val in enumerate(joint_pred):
                print(f"  joint_{i}: {val:.4f}")

            success = execute_pose_plan_with_seed(joint_pred, target_xyz, group_name)
            if not success:
                print("⚠️ 尝试失败，保留当前状态")

        except Exception as e:
            print("❌ 错误:", e)

if __name__ == '__main__':
    main()
