#!/usr/bin/env python3
from flask import Flask, request, jsonify
import threading
import rospy
import signal
import sys
import traceback
from moveit_commander import roscpp_initialize
from std_msgs.msg import Float32MultiArray
from auto_moveit_control import DualEEArmMover
from scipy.spatial.transform import Rotation as R
import numpy as np

app = Flask(__name__)
current_status = {'status': 'not_on_move'}
group_instances = {}  # 缓存不同 group 的控制器实例

def solve_grasp_quaternion(x, y, z, ee_link_name=""):
    target = np.array([x, y, z])
    
    if "left" in ee_link_name:
        elbow=np.array([-0.31, 0.237, 1.226])
        y_axis = elbow - target
        x_down = np.array([0, 0, -1])
    elif "right" in ee_link_name:
        elbow=np.array([0.31, 0.237, 1.226])
        y_axis = -elbow + target
        x_down = np.array([0, 0, 1])
    else:
        elbow=np.array([-0.31, 0.237, 1.226])
        y_axis = elbow - target
        x_down = np.array([0, 0, -1])

    y_axis /= np.linalg.norm(y_axis)

    if abs(np.dot(x_down, y_axis)) > 0.95:
        x_down = np.array([0, -1, 0])

    z_axis = np.cross(x_down, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    R_mat = np.column_stack((x_axis, y_axis, z_axis))
    rotation = R.from_matrix(R_mat)
    quat = rotation.as_quat()
    return quat

def get_mover(group_name):
    if group_name not in group_instances:
        print(f"[Info] 初始化 MoveIt 规划组：{group_name}")
        group_instances[group_name] = DualEEArmMover(group_name=group_name)
    return group_instances[group_name]

def get_ee_link(params, mover):
    ee_link = params.get('ee_link')
    if ee_link:
        return ee_link
    try:
        ee_link = mover.group.get_end_effector_link()
        if not ee_link:
            raise ValueError("group.get_end_effector_link() 返回为空")
        print(f"[Info] 自动获取 ee_link: {ee_link}")
        return ee_link
    except Exception as e:
        raise ValueError(f"无法从 group 获取末端执行器链接: {e}")

def run_motion_thread(func):
    def wrapped():
        current_status['status'] = 'on_move'
        try:
            func()
        except Exception as e:
            print(f"[Error] Motion thread exception: {e}")
            traceback.print_exc()
        current_status['status'] = 'not_on_move'
    threading.Thread(target=wrapped, daemon=True).start()

@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()
    print(f"[DEBUG] 收到请求: {data}")
    if not data or 'action' not in data or 'group' not in data:
        return jsonify(error="Missing 'action' or 'group' field"), 400

    group_name = data['group']
    action = data['action']
    params = data.get('params', {})
    if not isinstance(params, dict):
        return jsonify(error="'params' must be a dict"), 400

    mover = get_mover(group_name)

    try:
        ee_link = get_ee_link(params, mover)
    except Exception as e:
        return jsonify(error=f"ee_link error: {str(e)}"), 400
        
    if action == 'move_to':
        try:
            x = float(params['x'])
            y = float(params['y'])
            z = float(params['z'])
        except KeyError as e:
            return jsonify(error=f"Missing field: {e}"), 400

        # ===== 自动推断左右臂，决定 arm_flag =====
        if "left" in ee_link:
            arm_flag = 0
        elif "right" in ee_link:
            arm_flag = 1
        else:
            return jsonify(error="❌ 无法识别 ee_link 为左或右臂"), 400

        rospy.loginfo(f"[FastAPI] 准备发布位置请求: arm={arm_flag}, pos=({x:.3f}, {y:.3f}, {z:.3f})")

        # ===== 发布 ROS 话题 =====
        pub = rospy.Publisher("/iknet_position_input", Float32MultiArray, queue_size=1)
        msg = Float32MultiArray(data=[arm_flag, x, y, z])

        # 推荐稍微延时发布确保 subscriber 存在
        def motion():
            rospy.sleep(0.3)
            pub.publish(msg)
            rospy.loginfo("📤 已发布目标位置到 /iknet_position_input")

        run_motion_thread(motion)
        return jsonify(status='motion_started', action=action, group=group_name, ee_link=ee_link)

    elif action == 'move_to_pose':
        try:
            x = float(params['x'])
            y = float(params['y'])
            z = float(params['z'])
            qx = float(params['qx'])
            qy = float(params['qy'])
            qz = float(params['qz'])
            qw = float(params['qw'])
        except KeyError as e:
            return jsonify(error=f"Missing field: {e}"), 400

        def motion():
            mover.move_to_pose(x, y, z, qx, qy, qz, qw, ee_link=ee_link)

        run_motion_thread(motion)
        return jsonify(status='motion_started', action=action, group=group_name, ee_link=ee_link)

    elif action == 'get_current_pose':
        pose = mover.get_current_pose_info(ee_link=ee_link)
        return jsonify(status='done', action=action, group=group_name, ee_link=ee_link, pose={
            "position": {"x": pose[0], "y": pose[1], "z": pose[2]},
            "orientation": {"qx": pose[3][0], "qy": pose[3][1], "qz": pose[3][2], "qw": pose[3][3]}
        })

    elif action == 'shutdown':
        mover.shutdown()
        return jsonify(status='shutdown_complete')

    else:
        return jsonify(error=f"Unknown action: {action}"), 400

@app.route('/', methods=['GET'])
def query_status():
    action = request.args.get('action')
    if action == 'ask_status':
        return jsonify(status=current_status['status'])
    return jsonify(error="Unknown action"), 400

@app.errorhandler(500)
def internal_error(error):
    traceback.print_exc()
    return jsonify(error="Internal server error"), 500

def graceful_exit(signum, frame):
    print("\n🛑 收到 Ctrl+C，正在退出...")
    for mover in group_instances.values():
        mover.shutdown()
    print("✅ 已退出")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)

if __name__ == '__main__':
    roscpp_initialize([])
    rospy.init_node('flask_moveit_server', anonymous=True)
    print("✅ Flask+MoveIt 控制服务器启动：http://0.0.0.0:5000")
    print("🛑 按 Ctrl+C 退出")
    app.run(host='0.0.0.0', port=5000)
