import rclpy
import requests
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from my_yolo_cpp_pkg import detected_object_id

import websocket
import json
import math
import psutil  # CPU 사용량 측정을 위한 패키지 추가

object_name = {0: 'can', 1: 'paper', 2: 'plastic'}
class SpringBridge(Node):

    def __init__(self):
        super().__init__('spring_bridge')

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://192.168.0.58:8080/robot")

        # battery
        self.latest_battery = None
        self.subscription_battery = self.create_subscription(
            BatteryState,
            '/battery_state',
            self.battery_callback,
            10
        )
        self.create_timer(
            1.0,
            self.send_battery
        )

        # odom
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.create_timer(
            0.2,
            self.send_robot_pose
        )

        # object
        self.object_sub = self.create_subscription(
            detected_object_id.DetectedObject,
            '/detected_object_info',
            self.object_callback,
            10
        )

        # camera
        self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.camera_callback,
            10
        )

        self.create_timer(
            1.0,
            self.send_cpu_usage
        )

        self.create_timer(
            1.0,
            self.send_memory_usage
        )
        self.create_timer(
            1.0,
            self.send_disk_usage
        )



    def battery_callback(self, msg):
        self.latest_battery = msg.percentage

    def send_battery(self):
        if self.latest_battery is None:
            return

        data = {
            "type": "battery",
            "battery": self.latest_battery
        }

        self.ws.send(json.dumps(data))

        self.get_logger().info(str(data))


    def odom_callback(self, msg):
        data = {
            "type": "odom",
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "speed": msg.twist.twist.linear.x
        }
        self.ws.send(json.dumps(data))
        
        self.get_logger().info(str(data))

    def send_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation

            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            data = {
                "type": "robot_pose",
                "x": x,
                "y": y,
                "yaw": yaw
            }
            self.ws.send(json.dumps(data))

        except Exception as e:
            self.get_logger().warn(
                f"TF error: {e}"
            )

    def object_callback(self, msg):
        data = {
            "type": "detection",
            "object_name": object_name.get(msg.id, "-"),
            "confidence": msg.confidence
        }
        self.ws.send(json.dumps(data))

        self.get_logger().info(str(data))

    def camera_callback(self, msg):
        try:
            self.ws.send(msg.data, opcode=websocket.ABNF.OPCODE_BINARY)

        except Exception as e:
            self.get_logger().error(str(e))


    def send_cpu_usage(self):
        # interval=None으로 설정해야 노드가 멈추지(blocking) 않고 이전 측정 이후의 CPU 사용률을 바로 가져옵니다.
        cpu_percent = psutil.cpu_percent(interval=None)

        data = {
            "type": "cpu",
            "cpu_usage": cpu_percent
        }
        

        self.ws.send(json.dumps(data))
        self.get_logger().info(str(data))


    def send_memory_usage(self):
        # 메모리 사용률(%) 가져오기
        mem_percent = psutil.virtual_memory().percent

        data = {
            "type": "memory",
            "memory_usage": mem_percent
        }

        self.ws.send(json.dumps(data))
        self.get_logger().info(str(data))

    def send_disk_usage(self):
        # 루트 디렉터리('/') 기준 디스크 사용률(%) 가져오기
        disk_percent = psutil.disk_usage('/').percent

        data = {
            "type": "disk",
            "disk_usage": disk_percent
        }

        self.ws.send(json.dumps(data))
        self.get_logger().info(str(data))

def main():
    rclpy.init()
    node = SpringBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()