import rclpy
from rclpy.node import Node

import websocket
import json
import subprocess

from std_msgs.msg import String


class CommandBridge(Node):

    def __init__(self):
        super().__init__('command_bridge')

        # Spring WebSocket 연결
        self.ws = websocket.WebSocket()
        self.ws.connect("ws://192.168.0.58:8080/robot_command")

        self.should_shutdown = False

        self.get_logger().info(
            "Command Bridge Connected"
        )

        self.launch_process = None

        # WebSocket 수신 timer
        self.timer = self.create_timer(
            0.1,
            self.receive_command
        )

        self.cancel_pub = self.create_publisher(
            String,
            "/navigation_command",
            10
        )

        self.create_timer(
            1.0,
            self.check_launch_process
        )

        self.subscription_battery_low = self.create_subscription(
            String,
            '/battery_low',
            self.battery_callback,
            10
        )

    def check_launch_process(self):
        if self.launch_process is not None:
            if self.launch_process.poll() is not None:
                self.get_logger().info(
                    "Navigation launch process terminated"
                )
                self.launch_process = None

    def receive_command(self):
        try:
            self.ws.settimeout(0.01)
            data = self.ws.recv()

        except websocket.WebSocketTimeoutException:
            return

        except Exception:
            self.should_shutdown = True
            return

        if not data:
            return

        self.get_logger().info(
            f"Received : {data}"
        )

        try:
            msg_json = json.loads(data)
        except json.JSONDecodeError:
            return
        
        if msg_json.get("type") != "command":
            return

        command = msg_json.get("command")

        if command == "START":
            self.start_navigation()

        elif command == "STOP":
            self.stop_navigation()

    def is_auto_nav_alive(self):
        node_names = self.get_node_names()

        return any(
            name.strip('/') == 'auto_nav'
            for name in node_names
        )
    
    def start_navigation(self):
        if self.launch_process is not None:
            if self.launch_process.poll() is None:
                self.get_logger().warn(
                    "Navigation launch already running"
                )
                return
        if self.is_auto_nav_alive():
            self.get_logger().warn(
                "auto_nav already running"
            )
            return

        self.launch_process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "navigation",
                "navigation.launch.py"
            ],
            start_new_session=True
        )

        self.get_logger().info(
            "Launch started"
        )

    def stop_navigation(self):
        msg = String()
        msg.data = "STOP"

        self.cancel_pub.publish(msg)

    def battery_callback(self, msg):
        command_msg = String()
        command_msg.data = "BATTERY_LOW"

        self.cancel_pub.publish(command_msg)

def main():
    rclpy.init()
    node = CommandBridge()

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )
            
    except KeyboardInterrupt:
        node.get_logger().info("SIGINT 수신 - CommandBridge 종료")

    finally:
        try:
            node.ws.close()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()