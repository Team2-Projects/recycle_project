import rclpy
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf2_ros
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)

import websocket
import json
import math
import psutil 
import time

object_name = {0: 'can', 1: 'paper', 2: 'plastic', 3: 'trash', 4: 'person'}
class SpringBridge(Node):

    def __init__(self):
        super().__init__('spring_bridge')

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://192.168.0.58:8080/robot")

        self.should_shutdown = False

        # battery
        self.latest_battery = None
        self.last_sent_battery = None
        self.filtered_battery = None
        self.battery_low_alerted = False
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

        self.battery_low_pub = self.create_publisher(
            String,
            "/battery_low",
            10
        )

        # tf
        self.last_pose = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.create_timer(
            0.5,
            self.send_robot_pose
        )

        # path
        self.subscription_path = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        # object
        self.create_subscription(
            String,
            "object_found",
            self.object_callback,
            10
        )

        # recycle_success
        self.create_subscription(
            String,
            "recycle_success",
            self.recycle_success_callback,
            10
        )

        # camera
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            CompressedImage,
            "/yolo/image/compressed",
            self.camera_callback,
            camera_qos
        )

        # robotStatus 
        self.create_subscription( 
            String, 
            "/robot_status", 
            self.robot_status_callback, 
            10 
        )

        # robotTask
        self.create_subscription( 
            String, 
            "/robot_task", 
            self.robot_task_callback, 
            10 
        )

        # schedule_status
        self.create_subscription(
            String,
            "/schedule_status",
            self.schedule_status_callback,
            10
        )

        # voice
        self.subscription = self.create_subscription(
            String,
            'speech_to_text',
            self.listener_callback,
            10
        )

        # system
        psutil.cpu_percent(interval=None)
        self.create_timer(
            3.0,
            self.send_system_usage
        )

        self.status_check_count = 0

        self.status_timer = self.create_timer(
            0.5,
            self.send_current_robot_status
        )

    def is_auto_nav_alive(self):
        node_names = self.get_node_names()
        return (
            'auto_nav' in node_names or '/auto_nav' in node_names
        )

    def send_ws(self, data):
        try:
            self.ws.send(json.dumps(data))

        except Exception as e:
            self.get_logger().error(
                f"WebSocket send error: {e}"
            )
            self.get_logger().error(
                "웹 서버 연결이 끊겨 ROS 노드를 종료합니다."
            )

            self.should_shutdown = True

    def send_current_robot_status(self):
        self.status_check_count += 1

        if self.is_auto_nav_alive():

            self.send_ws({
                "type": "robot_status",
                "eventType": "state",
                "status": "Running"
            })

            self.destroy_timer(self.status_timer)
            return

        # 3번까지 확인
        if self.status_check_count >= 3:

            self.send_ws({
                "type": "robot_status",
                "eventType": "state",
                "status": "Stop"
            })

            self.destroy_timer(self.status_timer)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi

        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

    def battery_callback(self, msg):
        battery = msg.percentage
        alpha = 0.8

        if self.filtered_battery is None:
            self.filtered_battery = battery
        else:
            self.filtered_battery = (
                self.filtered_battery * alpha
                + battery * (1 - alpha)
            )

        self.latest_battery = self.filtered_battery

    def send_battery(self):
        if self.latest_battery is None:
            return

        if self.last_sent_battery is not None:
            diff = abs(self.latest_battery - self.last_sent_battery)
            if diff < 1.0:
                return
            if diff > 3.0:
                return

        self.last_sent_battery = self.latest_battery
        
        if self.last_sent_battery <= 30 and not self.battery_low_alerted:
            msg = String()
            msg.data = json.dumps({
                "battery": self.last_sent_battery,
                "status": "low"
            })
            self.battery_low_pub.publish(msg)

            self.battery_low_alerted = True

        elif self.last_sent_battery > 30:
            self.battery_low_alerted = False

        data = {
            "type": "battery",
            "battery": self.latest_battery
        }

        self.send_ws(data)

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

            if self.last_pose is not None:
                dx = abs(x - self.last_pose["x"])
                dy = abs(y - self.last_pose["y"])
                dyaw = abs(
                    self.normalize_angle(
                        yaw - self.last_pose["yaw"]
                    )
                )

                if dx < 0.02 and dy < 0.02 and dyaw < 0.05:
                    return

            data = {
                "type": "robot_pose",
                "x": x,
                "y": y,
                "yaw": yaw
            }
            self.last_pose = data
            self.send_ws(data)

        except Exception as e:
            self.get_logger().warn(
                f"TF error: {e}"
            )

    def path_callback(self, msg):
        try:
            path = []

            for pose_stamped in msg.poses:
                path.append({
                    "x": pose_stamped.pose.position.x,
                    "y": pose_stamped.pose.position.y
                })

            data = {
                "type": "navigation_path",
                "path": path
            }

            self.send_ws(data)

        except Exception as e:
            self.get_logger().error(
                f"path error: {e}"
            )

    def object_callback(self, msg):
        try:
            event = json.loads(msg.data)
            event["type"] = "detection"
            self.send_ws(event)

        except Exception as e:
            self.get_logger().error(
                f"object error: {e}"
            )

    def recycle_success_callback(self, msg):
        try:
            event = json.loads(msg.data)
            event["type"] = "recycleHistory"
            self.send_ws(event)

        except Exception as e:
            self.get_logger().error(
                f"recycle_success error: {e}"
            )

    def camera_callback(self, msg):
        try:
            self.ws.send(
                msg.data,
                opcode=websocket.ABNF.OPCODE_BINARY
            )

        except Exception as e:
            self.get_logger().error(
                f"Camera send error: {e}"
            )

    def robot_status_callback(self, msg): 
        try:
            event = json.loads(msg.data)
            event["type"] = "robot_status"

            self.send_ws(event)

        except Exception as e:
            self.get_logger().error(
                f"robot_status error: {e}"
            )

    def robot_task_callback(self, msg):
        try:
            event = json.loads(msg.data)
            event["type"] = "robot_task"
            self.send_ws(event)

        except Exception as e:
            self.get_logger().error(
                f"robot_task error: {e}"
            )
    
    def schedule_status_callback(self, msg):
        try:
            event = json.loads(msg.data)
            event["type"] = "schedule_status"

            self.send_ws(event)

        except Exception as e:
            self.get_logger().error(
                f"schedule_status error: {e}"
            )

    def listener_callback(self, msg):
        text = msg.data.strip()

        data = {
            "type": "voice_msg",
            "msg": text
        }
        self.send_ws(data)

    def send_system_usage(self):
        data = {
            "type": "system",
            "cpu": psutil.cpu_percent(interval=None),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }
        self.send_ws(data)


def main():
    rclpy.init()
    node = SpringBridge()

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


if __name__ == '__main__':
    main()