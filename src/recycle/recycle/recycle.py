import rclpy
from rclpy.node import Node

class Recycle(Node):  
    def __init__(self):
        super().__init__("recycle")

        self.get_logger().info("Recycle Start")


def main(args=None):
    rclpy.init(args=args)

    node = Recycle()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()