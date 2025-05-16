#!/usr/bin/env python
# -*- coding: utf-8 -*-

from auto_moveit_control import DualEEArmMover
from moveit_commander import roscpp_initialize
from moveit_commander import MoveGroupCommander
import rospy

roscpp_initialize([])
rospy.init_node("dual_ee_mover_test", anonymous=True)

# 用户选择 group_name
group_name = input("请输入控制的 MoveIt 规划组名（如 left_arm、right_arm、full）: ").strip()
if not group_name:
    group_name = "full"
    print("[Info] 未输入，默认使用 group: full")

try:
    group_commander = MoveGroupCommander(group_name)
    ee_link = group_commander.get_end_effector_link()
    if not ee_link:
        raise ValueError("[警告] 未配置默认末端执行器，将使用 fallback")
except Exception as e:
    print(f"[警告] 获取默认末端执行器失败：{e}")
    # fallback
    if "left" in group_name:
        ee_link = "left_endeffector_center_link"
    elif "right" in group_name:
        ee_link = "right_endeffector_center_link"
    else:
        ee_link = "right_wrist_link"

print(f"[Info] 使用 group: {group_name}, 默认末端执行器: {ee_link}")
mover = DualEEArmMover(group_name=group_name)

# 自动推导默认 offset
DEFAULT_OFFSET = [0.0, -0.2, 0.025] if "left" in ee_link else [0.0, 0.2, 0.025]
DEFAULT_TOL = 0.7
DEFAULT_WEIGHT = 0.5

def get_xyz_quat():
    x = float(input("目标 x: "))
    y = float(input("目标 y: "))
    z = float(input("目标 z: "))
    qx = float(input("姿态 qx: "))
    qy = float(input("姿态 qy: "))
    qz = float(input("姿态 qz: "))
    qw = float(input("姿态 qw: "))
    return x, y, z, qx, qy, qz, qw

while True:
    print("\n==== 功能菜单 ====")
    print("1. 查询当前位置")
    print("2. 移动到指定四元数位置（含姿态容差）")
    print("3. 移动到预设位置（命名目标）")
    print("4. 只移动到目标位置（自动姿态）")
    print("5. ✅ 工具尖端反推：带 offset + 姿态容差")
    print("q. 退出")

    choice = input("请输入操作编号: ").strip().lower()

    try:
        if choice == '1':
            pose = mover.get_current_pose_info(ee_link=ee_link)
            print(f"[Pose] 位置: {pose[0]}  姿态: {pose[3]}")

        elif choice == '2':
            x, y, z, qx, qy, qz, qw = get_xyz_quat()
            mover.move_with_orientation_tolerance(
                x, y, z, qx, qy, qz, qw,
                tolerance_roll=DEFAULT_TOL,
                tolerance_pitch=DEFAULT_TOL,
                tolerance_yaw=DEFAULT_TOL,
                weight=DEFAULT_WEIGHT,
                ee_link=ee_link
            )

        elif choice == '3':
            name = input("请输入命名目标（如 leftEE_open）: ").strip()
            mover.move_to_named_target(name)

        elif choice == '4':
            x = float(input("目标 x: "))
            y = float(input("目标 y: "))
            z = float(input("目标 z: "))
            mover.move_to_position_only(x, y, z, ee_link=ee_link)

        elif choice == '5':
            x, y, z, qx, qy, qz, qw = get_xyz_quat()
            print(f"✅ 默认偏移: {DEFAULT_OFFSET}，容差: {DEFAULT_TOL:.2f} rad (~20°)")
            mover.move_tip_to_target_with_offset(
                x, y, z, qx, qy, qz, qw,
                offset_vec=DEFAULT_OFFSET,
                tol_roll=DEFAULT_TOL,
                tol_pitch=DEFAULT_TOL,
                tol_yaw=DEFAULT_TOL,
                weight=DEFAULT_WEIGHT,
                ee_link=ee_link
            )

        elif choice == 'q':
            break
        else:
            print("[提示] 无效输入")
    except ValueError:
        print("[错误] 输入格式不正确")

mover.shutdown()
print("✅ 已退出测试程序。")
