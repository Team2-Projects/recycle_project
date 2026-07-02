import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np


clf_idx = {'can': 0, 'paper': 1, 'plastic': 2}


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.frame_count = 0
        self.bridge = CvBridge() # ★ bridge 초기화도 잊지 마세요 ★
        self.declare_parameter('conf', 0.5)
        # 모델 경로를 확인하세요
        self.model = YOLO('/home/hee/turtlebot3_ws/src/my_yolo_cpp_pkg/models/yolo_8n_trained_1_openvino_model')
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.listener_callback, 10)

        # self.subscription = self.create_subscription(
        #     Image, '/image_raw', self.listener_callback, 10)

  

    def listener_callback(self, msg):
        self.frame_count += 1

        if self.frame_count % 5 != 0:
            return

        else:
            conf_threshold = self.get_parameter('conf').get_parameter_value().double_value
            #transformed by 8byte int 
            np_arr = np.frombuffer(msg.data, np.uint8)
            #reconstruction img

            # 1. CvBridge를 사용하여 ROS 이미지를 OpenCV(BGR)로 변환
            # frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=1.5, beta=30)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # print(f"이미지 형태 (Shape): {frame_bgr.shape}")
            # # 추론
            results = self.model.predict(source=frame_bgr, imgsz=640, conf=conf_threshold, verbose=False)
            
            res = results[0]
        
            if len(res.boxes) > 0:
                  # 2. 모든 객체의 신뢰도(conf)를 가져와서 가장 높은 인덱스를 찾음
                  # res.boxes.conf는 텐서 형태이므로 리스트로 변환 후 사용
                confidences = res.boxes.conf.tolist()
                max_conf_idx = confidences.index(max(confidences))

                  # 3. 가장 높은 신뢰도를 가진 객체의 클래스 ID 추출
                best_cls_id = int(res.boxes.cls[max_conf_idx].item())
                best_name = res.names[best_cls_id]
                best_idx = clf_idx[best_name]
                coord = res.boxes.xywh[max_conf_idx].tolist()

                x,y,w,h = coord
              # print(x,y,w,h)
                pt1_x = int(x - (w/2))
                pt1_y = int(y - (h/2))
                pt2_x = int(x + (w/2))
                pt2_y = int(y + (h/2))

                org_x = pt1_x - 5
                org_y = pt1_y - 5
                  
                cv2.rectangle(frame_bgr, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 40, 200), 3)
                cv2.putText(frame_bgr, best_name, (org_x, org_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, thickness = 3, color = (255, 0, 0))
                cv2.imshow("YOLO Python Node", frame_bgr)
                # cv2.imshow("img", frame)
                cv2.waitKey(10)



            else:
                best_name = None
                best_idx = None
                coord = None

                cv2.imshow("YOLO Python Node", frame_bgr)
                cv2.waitKey(10)
              # res_plotted_rgb = res.plot()
              # res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)
              # cv2_imshow(res_plotted_bgr)

              # cv2_imshow(frame_bgr)
            
# --------------

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()