import requests
import json

server_url = "http://localhost:5000"  # 根据你的服务器地址改
group_name = "left_arm"  # 默认测试组，可以改成 left_arm / right_arm 等

def send_post(action, params=None):
    payload = {
        "action": action,
        "group": group_name,
        "params": params or {}
    }
    try:
        response = requests.post(server_url, json=payload)
        print(f"[Server Response] {response.status_code}: {response.json()}\n")
    except Exception as e:
        print(f"[Error] Failed to send POST: {e}")

def send_get(action):
    try:
        response = requests.get(server_url, params={"action": action})
        print(f"[Server Response] {response.status_code}: {response.json()}\n")
    except Exception as e:
        print(f"[Error] Failed to send GET: {e}")

def main_menu():
    while True:
        print("\n===== 测试菜单 =====")
        print("1. move_to 位置移动 (只位置)")
        print("2. move_to_pose 位姿移动 (位置+姿态)")
        print("3. move_tip_to_target_with_offset (自动offset，末端对齐)")
        print("4. move_to_named (命名姿态移动)")
        print("5. rotate (绕base_link旋转)")
        print("6. get_current_pose (查询当前位姿)")
        print("7. shutdown (关闭机器人)")
        print("q. 退出")
        choice = input("请输入选择: ").strip().lower()

        if choice == '1':
            x = float(input("x: "))
            y = float(input("y: "))
            z = float(input("z: "))
            send_post("move_to", {"x": x, "y": y, "z": z})

        elif choice == '2':
            x = float(input("x: "))
            y = float(input("y: "))
            z = float(input("z: "))
            qx = float(input("qx: "))
            qy = float(input("qy: "))
            qz = float(input("qz: "))
            qw = float(input("qw: "))
            send_post("move_to_pose", {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw})

        elif choice == '3':
            x = float(input("x: "))
            y = float(input("y: "))
            z = float(input("z: "))
            qx = float(input("qx: "))
            qy = float(input("qy: "))
            qz = float(input("qz: "))
            qw = float(input("qw: "))
            # offset_vec 如果不填就自动推断，也可以手动加上：
            use_custom_offset = input("自定义offset? (y/n): ").strip().lower()
            params = {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}
            if use_custom_offset == 'y':
                dx = float(input("offset dx: "))
                dy = float(input("offset dy: "))
                dz = float(input("offset dz: "))
                params["offset_vec"] = [dx, dy, dz]
            send_post("move_tip_to_target_with_offset", params)

        elif choice == '4':
            name = input("请输入命名目标名 (如 'home', 'open', 'close' 等): ").strip()
            send_post("move_to_named", {"name": name})

        elif choice == '5':
            axis = input("旋转轴 (x/y/z): ").strip().lower()
            delta_deg = float(input("旋转角度（度）: "))
            send_post("rotate", {"axis": axis, "delta_deg": delta_deg})

        elif choice == '6':
            send_post("get_current_pose")

        elif choice == '7':
            confirm = input("确定要shutdown？(y/n): ").strip().lower()
            if confirm == 'y':
                send_post("shutdown")

        elif choice == 'q':
            print("✅ 测试结束")
            break

        else:
            print("❌ 无效选择，请重新输入。")

if __name__ == "__main__":
    print(f"🎯 连接到服务器：{server_url}，默认组：{group_name}")
    main_menu()
