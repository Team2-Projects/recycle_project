from robot_position import RobotPosition


categories = ["일반", "캔", "플라", "유리", "종이"]


class IntentParser:

    def __init__(self):
        # RobotPosition 객체 생성
        self.robot_position = RobotPosition()

    def parse(self, text: str) -> str:

        if "정지" in text:
            return "정지가 완료되었습니다."

        elif "속도" in text and "증가" in text:
            return "더 빠르게 주행을 시작합니다."

        elif "속도" in text and "감소" in text:
            return "더 느리게 주행을 시작합니다."

        # ==========================================
        # 현재 로봇 위치 확인
        # ==========================================
        elif "현재위치" in text or "현재 위치":

            x, y = self.robot_position.get_position()

            if x is None or y is None:
                return "현재 로봇 위치를 확인할 수 없습니다."

            return f"현재 로봇 위치는 동쪽으로 {x:.2f}미터,  북쪽으로 {y:.2f}미터 떨어져 있습니다."


        elif "분리수거장" in text:

            x, y = self.robot_position.get_position()
        
            del_x = x - (-0.2)
            del_y = y - (-1.5)
            if x is None or y is None:
                return "현재 로봇 위치를 확인할 수 없습니다."
            
            if del_x >= 0 or del_y >= 0
                return f"분리수거장은는 동쪽으로 {del_x:.2f}미터, 북쪽으로 {del_y:.2f}미터 떨어져 있습니다."

            elif del_x < 0 or del_y >= 0
                return f"분리수거장은는 서쪽으로 {del_x:.2f}미터, 북쪽으로 {del_y:.2f}미터 떨어져 있습니다."

            elif del_x >= 0 or del_y < 0
                return f"분리수거장은는 동쪽으로 {del_x:.2f}미터, 남쪽으로 {del_y:.2f}미터 떨어져 있습니다."

            elif del_x < 0 or del_y < 0
                return f"분리수거장은는 서쪽으로 {del_x:.2f}미터, 남쪽으로 {del_y:.2f}미터 떨어져 있습니다."

        # ==========================================
        # 쓰레기 종류
        # ==========================================
        elif any(category in text for category in categories):

            for category in categories:

                if category in text:

                    if category == "일반":
                        return f"{category} 쓰레기를 수거하겠습니다."

                    elif category == "캔":
                        return f"{category} 쓰레기를 수거하겠습니다."

                    elif category == "플라":
                        return f"{category}스틱 쓰레기를 수거하겠습니다."

                    elif category == "유리":
                        return f"{category}병 쓰레기를 수거하겠습니다."

                    elif category == "종이":
                        return f"{category} 쓰레기를 수거하겠습니다."

        return text