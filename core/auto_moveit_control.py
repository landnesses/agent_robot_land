from moveit_commander import MoveGroupCommander, roscpp_shutdown
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, OrientationConstraint ,JointConstraint
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

class DualEEArmMover:
    def __init__(self, group_name):
        self.group = MoveGroupCommander(group_name)
        self.group.set_planning_time(0.3)
        self.group.set_num_planning_attempts(10)
        self.group.allow_replanning(True)
        
        default_ee_link = self.group.get_end_effector_link()
        if not default_ee_link:
            raise ValueError(f"组 {group_name} 无法确定末端执行器")
        self.default_ee_link = default_ee_link

    
    def get_current_pose_info(self, ee_link=None):
        ee_link = ee_link or self.default_ee_link
        pose = self.group.get_current_pose(end_effector_link=ee_link).pose
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        r = R.from_quat([qx, qy, qz, qw])
        roll_deg, pitch_deg, yaw_deg = r.as_euler('xyz', degrees=True)

        print(f"\n[当前位置 @ {ee_link}] x={x:.4f}, y={y:.4f}, z={z:.4f}")
        print(f"[当前姿态四元数] qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}, qw={qw:.4f}")
        print(f"[当前姿态角度] roll={roll_deg:.2f}°, pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°")
        return x, y, z, [qx, qy, qz, qw]

    def move_to_position_only(self, x, y, z, ee_link=None):
        ee_link = ee_link or self.default_ee_link

        self.group.set_start_state_to_current_state()
        self.group.clear_pose_targets()  # 清除姿态目标，避免干扰

        # 设置纯位置目标（默认对当前 EE link）
        self.group.set_position_target([x, y, z], end_effector_link=ee_link)

        success, plan = self._plan_and_execute()
        self.group.clear_pose_targets()
        return success


    def move_with_orientation_tolerance(self, x, y, z, qx, qy, qz, qw,
                                        tolerance_roll=1.5, tolerance_pitch=1.5, tolerance_yaw=1.5,
                                        weight=1.0, ee_link=None):
        ee_link = ee_link or self.default_ee_link

        oc = OrientationConstraint()
        oc.link_name = ee_link
        oc.header.frame_id = self.group.get_planning_frame()
        oc.orientation.x = qx
        oc.orientation.y = qy
        oc.orientation.z = qz
        oc.orientation.w = qw
        oc.absolute_x_axis_tolerance = tolerance_roll
        oc.absolute_y_axis_tolerance = tolerance_pitch
        oc.absolute_z_axis_tolerance = tolerance_yaw
        oc.weight = weight

        constraints = Constraints()
        constraints.orientation_constraints.append(oc)
        self.group.set_path_constraints(constraints)

        pose_target = Pose()
        pose_target.position.x = x
        pose_target.position.y = y
        pose_target.position.z = z
        pose_target.orientation.x = qx
        pose_target.orientation.y = qy
        pose_target.orientation.z = qz
        pose_target.orientation.w = qw

        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(pose_target, end_effector_link=ee_link)

        success, plan = self._plan_and_execute()
        self.group.clear_path_constraints()
        self.group.clear_pose_targets()
        return success

    def move_tip_to_target_with_offset(self, x, y, z, qx, qy, qz, qw, offset_vec,
                                       tol_roll=0.2, tol_pitch=0.2, tol_yaw=0.2, weight=1.0,
                                       ee_link=None):
        ee_link = ee_link or self.default_ee_link

        def pose_to_matrix(pos, quat):
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            return T

        T_tip = pose_to_matrix([x, y, z], [qx, qy, qz, qw])
        T_offset = np.eye(4)
        T_offset[:3, 3] = offset_vec
        T_wrist = T_tip @ np.linalg.inv(T_offset)

        pos = T_wrist[:3, 3]
        quat = R.from_matrix(T_wrist[:3, :3]).as_quat()

        print(f"[{ee_link}] Wrist pose after offset: pos={pos}, quat={quat}")
        return self.move_with_orientation_tolerance(
            pos[0], pos[1], pos[2],
            quat[0], quat[1], quat[2], quat[3],
            tolerance_roll=tol_roll,
            tolerance_pitch=tol_pitch,
            tolerance_yaw=tol_yaw,
            weight=weight,
            ee_link=ee_link
        )

    def move_to_named_target(self, name):
        self.group.set_start_state_to_current_state()
        self.group.set_named_target(name)
        success, plan = self._plan_and_execute()
        self.group.clear_pose_targets()
        return success

    def _plan_and_execute(self):
        plan_result = self.group.plan()
        if isinstance(plan_result, tuple):
            success, plan = bool(plan_result[0]), plan_result[1]
        else:
            success, plan = bool(plan_result), plan_result

        if success and plan:
            print("[ArmMover] Plan success, executing...")
            self.group.execute(plan, wait=True)
        else:
            print("[ArmMover] Planning failed.")
        return success, plan


    def shutdown(self):
        roscpp_shutdown()

