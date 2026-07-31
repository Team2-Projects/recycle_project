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

object_name = {0: 'can', 1: 'paper', 2: 'plastic'}
class SpringBridge(Node):

    def __init__(self):
        super().__init__('spring_bridge')

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://192.168.0.16:8080/robot")

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




def main():
    rclpy.init()
    node = SpringBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()