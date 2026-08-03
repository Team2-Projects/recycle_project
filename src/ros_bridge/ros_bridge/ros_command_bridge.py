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
        self.ws.connect("ws://192.168.0.16:8080/robot_command")

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

    def receive_command(self):
        try:
            # WebSocket 데이터 확인
            self.ws.settimeout(0.01)
            data = self.ws.recv()
            if not data:
                return

            self.get_logger().info(
                f"Received : {data}"
            )

            msg_json = json.loads(data)

            if msg_json["type"] == "command":

                command = msg_json["command"]
                
                self.get_logger().info(
                    f"Publish command : {command}"
                )
                if command == "START":
                    self.start_navigation()
                elif command == "STOP":
                    self.stop_navigation()

        except websocket.WebSocketTimeoutException:
            pass

        except Exception as e:
            self.get_logger().error(
                str(e)
            )

    def start_navigation(self):
        if self.launch_process is not None:
            self.get_logger().info(
                "Already running"
            )
            return

        self.launch_process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "navigation",
                "navigation.launch.py"
            ]
        )

        self.get_logger().info(
            "Launch started"
        )


    def stop_navigation(self):
        msg = String()
        msg.data = "STOP"

        self.cancel_pub.publish(msg)



def main():
    rclpy.init()
    node = CommandBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()