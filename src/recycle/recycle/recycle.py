import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
import time

class Recycle(Node):  
    def __init__(self):
        super().__init__("recycle")

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.move_backward()

        self.get_logger().info("Recycle Start")

    def move_backward(self):
        msg = Twist()
        msg.linear.x = -0.2
        msg.angular.z = 0.0

        start_time = time.time()
        while time.time() - start_time < 2.0:
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.1)
        
        self.stop_robot()

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)

    node = Recycle()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()