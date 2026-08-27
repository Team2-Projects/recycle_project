import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class RobotPosition(Node):

    def __init__(self):
        super().__init__('robot_position')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

    def get_position(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            return x, y

        
        except Exception as e:
            self.get_logger().warn(
                f"로봇 위치를 가져올 수 없습니다: {e}"
            )

            return None, None