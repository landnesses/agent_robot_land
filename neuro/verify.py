# extract_tf_chain.py（自动解析 TF 树 → 提取 link 间变换用于 DH 建模）

#!/usr/bin/env python3
import rospy
import tf
from urdf_parser_py.urdf import URDF

def get_link_chain(robot, start_link, end_link):
    chain = []
    link = end_link
    while link != start_link:
        joint = robot.parent_map.get(link)
        if not joint:
            raise ValueError(f"Link '{link}' is not connected to '{start_link}'")
        joint = joint[0]
        parent_link = robot.joint_map[joint].parent
        chain.append((parent_link, link))
        link = parent_link
    chain.reverse()
    return chain

def extract_tf_chain(start, end):
    rospy.init_node("extract_tf_chain", anonymous=True)
    listener = tf.TransformListener()
    robot = URDF.from_parameter_server()

    rospy.loginfo(f"🔍 查找 TF 路径：{start} → {end}")
    chain = get_link_chain(robot, start, end)

    rospy.loginfo(f"✅ 路径总共 {len(chain)} 段关节")
    rospy.loginfo(f"{'FROM':<25} → {'TO':<25}")
    for a, b in chain:
        rospy.loginfo(f"{a:<25} → {b:<25}")

    print(f"\n🧾 相对变换（translation + RPY）:")
    print(f"{'FROM':<25} {'TO':<25} {'tx':>7} {'ty':>7} {'tz':>7}  {'roll':>7} {'pitch':>7} {'yaw':>7}")

    rate = rospy.Rate(10)
    tf_ready = False
    while not tf_ready and not rospy.is_shutdown():
        try:
            for from_link, to_link in chain:
                listener.waitForTransform(from_link, to_link, rospy.Time(0), rospy.Duration(3.0))
            tf_ready = True
        except:
            rospy.logwarn("等待 TF 树可用中...")
            rate.sleep()

    for from_link, to_link in chain:
        try:
            (trans, rot) = listener.lookupTransform(from_link, to_link, rospy.Time(0))
            rpy = tf.transformations.euler_from_quaternion(rot)
            print(f"{from_link:<25} {to_link:<25} "
                  f"{trans[0]:7.3f} {trans[1]:7.3f} {trans[2]:7.3f}  "
                  f"{rpy[0]:7.3f} {rpy[1]:7.3f} {rpy[2]:7.3f}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"❌ 获取 {from_link} → {to_link} 失败")
            continue

if __name__ == "__main__":
    # ✅ 设置起点和终点 link 名称
    START_LINK = "waist_side_link"
    END_LINK = "left_endeffector_center_link"
    extract_tf_chain(START_LINK, END_LINK)
