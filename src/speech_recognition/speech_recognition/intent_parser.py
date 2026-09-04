from .robot_position import RobotPosition
import joblib
import numpy as np


NUMBER_MAP = {
    # 1
    "일": 1,
    "한": 1,
    "하나": 1,

    # 2
    "이": 2,
    "두": 2,
    "둘": 2,

    # 3
    "삼": 3,
    "세": 3,
    "셋": 3,

    # 4
    "사": 4,
    "네": 4,
    "넷": 4,

    # 5
    "오": 5,
    "다섯": 5,

    # 6
    "육": 6,
    "여섯": 6,

    # 7
    "칠": 7,
    "일곱": 7,

    # 8
    "팔": 8,
    "여덟": 8,

    # 9
    "구": 9,
    "아홉": 9,

    # 10
    "십": 10,
    "열": 10,

    # 11
    "십일": 11,
    "열한": 11,

    # 12
    "십이": 12,
    "열두": 12
}


Korean_map = {
    0: "영",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구",
}

number_indexs = np.arange(1, 13, 1)


class IntentParser:

    def __init__(self):
        self.robot_position = RobotPosition()

        self.text_recognition_model = joblib.load(
            "/home/hee/turtlebot3_ws/src/speech_recognition/models/text_recognition_model.joblib"
        )

        self.command_flag = -1

    def return_flag(self, text):
        self.command_flag = self.text_recognition_model.predict([text])[0]
        return self.command_flag

    def get_patrol_indexs(self, text):
        extracted_text = text
        patrol_indexs = []

        for word in extracted_text:

            # 한글 숫자인 경우
            if word in NUMBER_MAP:
                patrol_indexs.append(NUMBER_MAP[word] - 1)

            # 숫자 문자열인 경우
            else:
                try:
                    number = int(word)

                    if number in number_indexs:
                        patrol_indexs.append(number - 1)

                except ValueError:
                    pass

        patrol_indexs.append(6)

        return patrol_indexs

    def get_start_time(self, extracted_text):

        for word in extracted_text:

            # 한글 숫자인 경우
            if word in NUMBER_MAP:
                start_time = NUMBER_MAP[word]
                return start_time

            # 숫자 문자열인 경우
            else:
                try:
                    number = int(word)

                    if number in number_indexs:
                        start_time = number
                        return start_time

                except ValueError:
                    pass

        return 0

    def number_to_korean(self, text):

        return_text = ""

        for idx, word in enumerate(text):

            if word == "." and idx != len(text) - 1:
                return_text += "점"

            else:
                try:
                    number = int(word)

                    if number in Korean_map:
                        return_text += Korean_map[number]

                except ValueError:
                    return_text += word

        return return_text

    def parse(self):

        x, y = self.robot_position.get_position()

        if x is None or y is None:
            return "현재 로봇 위치를 확인할 수 없습니다."

        del_x = x - (-0.2)
        del_y = y - (-1.5)

        if del_x >= 0 and del_y >= 0:
            return_value = (
                f"분리수거장은 동쪽으로 {del_x:.2f}미터, "
                f"북쪽으로 {del_y:.2f}미터 떨어져 있습니다."
            )

        elif del_x < 0 and del_y >= 0:
            return_value = (
                f"분리수거장은 서쪽으로 {abs(del_x):.2f}미터, "
                f"북쪽으로 {del_y:.2f}미터 떨어져 있습니다."
            )

        elif del_x >= 0 and del_y < 0:
            return_value = (
                f"분리수거장은 동쪽으로 {del_x:.2f}미터, "
                f"남쪽으로 {abs(del_y):.2f}미터 떨어져 있습니다."
            )

        elif del_x < 0 and del_y < 0:
            return_value = (
                f"분리수거장은 서쪽으로 {abs(del_x):.2f}미터, "
                f"남쪽으로 {abs(del_y):.2f}미터 떨어져 있습니다."
            )

        return_value = self.number_to_korean(return_value)

        return return_value

       