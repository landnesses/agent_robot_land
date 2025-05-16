#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState

# 缓存结构：每组 old/new 对应多帧数据
frame_cache = {
    "left":  {"old": [], "new": []},
    "right": {"old": [], "new": []},
    "waist": {"old": [], "new": []}
}

frame_limit = 5  # 每组达到这个帧数后进行比对

def try_compare_all():
    ready = all(
        len(frame_cache[group]["old"]) >= frame_limit and
        len(frame_cache[group]["new"]) >= frame_limit
        for group in ["left", "right", "waist"]
    )
    if ready:
        print("\n=== 所有模块已收集到 5 帧，开始对比 ===\n")
        for group in ["left", "right", "waist"]:
            for i in range(frame_limit):
                compare_frame(group, i)
        rospy.signal_shutdown("对比完成")

def compare_frame(group, i):
    old = frame_cache[group]["old"][i]
    new = frame_cache[group]["new"][i]

    if not old.position or not new.position:
        print(f"[跳过] {group} 第 {i} 帧 position 为空")
        return
    if len(old.position) != len(new.position):
        print(f"[跳过] {group} 第 {i} 帧 joint 数不一致")
        return

    print(f"\n[{group.upper()}] 第 {i+1} 帧")
    for j, (o, n) in enumerate(zip(old.position, new.position)):
        diff = abs(o - n)
        status = "✅" if diff < 1e-5 else f"⚠️ 差异 {diff:.5f}"
        print(f"  关节{j:2d}: old={o: .5f}, new={n: .5f}  --> {status}")

# 各话题回调：简单追加进缓存
def left_old_cb(msg):  frame_cache["left"]["old"].append(msg); try_compare_all()
def left_new_cb(msg):  frame_cache["left"]["new"].append(msg); try_compare_all()
def right_old_cb(msg): frame_cache["right"]["old"].append(msg); try_compare_all()
def right_new_cb(msg): frame_cache["right"]["new"].append(msg); try_compare_all()
def waist_old_cb(msg): frame_cache["waist"]["old"].append(msg); try_compare_all()
def waist_new_cb(msg): frame_cache["waist"]["new"].append(msg); try_compare_all()

def main():
    rospy.init_node("joint_state_frame_compare")
    rospy.Subscriber("/left_arm_control", JointState, left_old_cb)
    rospy.Subscriber("/right_arm_control", JointState, right_old_cb)
    rospy.Subscriber("/waist_control", JointState, waist_old_cb)
    rospy.Subscriber("/left_arm_control_test", JointState, left_new_cb)
    rospy.Subscriber("/right_arm_control_test", JointState, right_new_cb)
    rospy.Subscriber("/waist_control_test", JointState, waist_new_cb)
    print("开始缓存每帧 joint state，等每组到达 5 帧后自动对比...")
    rospy.spin()

if __name__ == "__main__":
    main()
