import rclpy
import requests
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy
)

from my_yolo_cpp_pkg import detected_object_id

import websocket
import json

object_name = {0: 'can', 1: 'paper', 2: 'plastic'}
class SpringBridge(Node):

    def __init__(self):
        super().__init__('spring_bridge')

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://192.168.0.16:8080/robot")

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

        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.object_sub = self.create_subscription(
            detected_object_id.DetectedObject,
            '/detected_object_info',
            self.object_callback,
            10
        )

        self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.camera_callback,
            10
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos
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

    def map_callback(self, msg):
        data = {
            "type": "map",
            "width": msg.info.width,
            "height": msg.info.height,
            "resolution": msg.info.resolution,
            "origin_x": msg.info.origin.position.x,
            "origin_y": msg.info.origin.position.y,
            "map": list(msg.data)
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