#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iknet_moveit_position_only_refine.py
------------------------------------
神经网络 seed + refine + 单次执行 + 转发 result 到 relay 话题
"""

import rospy
import torch
import torch.nn as nn
import joblib
import numpy as np
import moveit_commander
from math import radians
from moveit_msgs.msg import MoveGroupActionResult
import actionlib_msgs.msg
import std_msgs.msg


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
    group  = moveit_commander.MoveGroupCommander(group_name)
    robot  = moveit_commander.RobotCommander()
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


# =======================  3. 执行轨迹 + 结果中继 ====================== #
def execute_joint_plan_and_record_result(joint_values, group_name="left_arm_full"):
    group  = moveit_commander.MoveGroupCommander(group_name)
    robot  = moveit_commander.RobotCommander()
    names  = group.get_active_joints()

    # ---- 限位裁剪 ---- #
    m = radians(0)
    for i, n in enumerate(names):
        joint = robot.get_joint(n)
        lo, hi = joint.bounds()
        joint_values[i] = float(np.clip(joint_values[i], lo + m, hi - m))

    group.set_start_state_to_current_state()
    group.set_joint_value_target(dict(zip(names, joint_values)))

    rospy.loginfo("🚀 正在执行轨迹 ...")
    success = group.go(wait=True)
    group.stop()

    if success:
        rospy.loginfo("✅ 轨迹执行完成")
    else:
        rospy.logerr("❌ 轨迹执行失败")

    return success



# =======================  4. 主程序入口  ====================== #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ 当前推理设备:", device)

    model = IKNet().to(device)
    model.load_state_dict(torch.load("iknet_model_weights.pt", map_location=device))
    model.eval()
    scaler_y = joblib.load("iknet_scaler_y.pkl")

    moveit_commander.roscpp_initialize([])
    rospy.init_node("iknet_position_only_demo", anonymous=True)

    print("🧪 输入: arm_flag x y z   (arm_flag=0 左臂, 1 右臂)")

    while not rospy.is_shutdown():
        try:
            inp = input("➤ 目标: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if inp.lower() in {"exit", "quit"}:
            break

        try:
            vals = list(map(float, inp.split()))
            if len(vals) != 4:
                print("❌ 必须输入 4 个值")
                continue
            arm_flag = int(vals[0])
            tgt_xyz  = vals[1:]
        except ValueError:
            print("❌ 输入格式错误")
            continue

        group_name = "left_arm_full" if arm_flag == 0 else "right_arm_full"

        # ---- Step 1: NN 推理 ---- #
        nn_in = torch.tensor([vals], dtype=torch.float32).to(device)
        with torch.no_grad():
            nn_out = model(nn_in).cpu().numpy()
        joint_seed = scaler_y.inverse_transform(nn_out)[0]
        print("🧠 NN 关节角(rad):", np.round(joint_seed, 4))

        # ---- Step 2: refine 到目标点 ---- #
        refined = refine_position_only_with_seed(joint_seed, tgt_xyz, group_name=group_name)
        print("🎯 refine 结果(rad):", np.round(refined, 4))

        # ---- Step 3: 执行 refine 轨迹 + result 转发 ---- #
        execute_joint_plan_and_record_result(refined, group_name=group_name)


if __name__ == "__main__":
    try:
        main()
    finally:
        moveit_commander.roscpp_shutdown()
