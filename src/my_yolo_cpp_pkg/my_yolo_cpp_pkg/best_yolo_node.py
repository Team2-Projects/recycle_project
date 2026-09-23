"""제어용 인식 노드가 발행한 검출 영상을 로컬 창에 표시한다."""

import cv2

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CompressedImage


WINDOW_NAME = 'YOLO Python Node'


class YoloImageViewer(Node):
    """모델을 실행하지 않고 웹과 같은 결과 영상을 구독한다."""

    def __init__(self):
        """오래된 영상이 쌓이지 않도록 최신 영상 하나만 받는다."""
        super().__init__('yolo_visualization_node')
        self.latest_frame = None
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.subscription = self.create_subscription(
            CompressedImage, '/yolo/image/compressed', self.listener_callback, image_qos)

    def listener_callback(self, msg):
        """공유된 JPEG 영상을 디코딩하고 손상된 영상은 건너뛴다."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except cv2.error:
            return
        if frame is not None:
            self.latest_frame = frame


def main(args=None):
    """Launch와 함께 창을 열고, 영상 수신이 없어도 창 이벤트를 처리한다."""
    rclpy.init(args=args)
    node = YoloImageViewer()
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        waiting = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(waiting, 'Waiting for detection images...', (25, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, waiting)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.latest_frame is not None:
                cv2.imshow(WINDOW_NAME, node.latest_frame)
                node.latest_frame = None
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')) or cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except cv2.error as exc:
        node.get_logger().error(f'검출 영상 창 오류: {exc}')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
