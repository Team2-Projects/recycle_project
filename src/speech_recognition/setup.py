import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'speech_recognition'

# OpenVINO 모델 파일 자동 등록
model_files = glob('models/whisper_tiny_openvino/*')

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

if model_files:
    data_files.extend([
        (
            os.path.join(
                'share',
                package_name,
                'models',
                'whisper_tiny_openvino'
            ),
            model_files
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob('launch/*.launch.py')
        )
    ])

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'my_interfaces'],
    zip_safe=True,
    maintainer='hee',
    maintainer_email='tmdgml88888@naver.com',
    description='STT Speech Recognition Package using OpenVINO',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tts_publisher_node = speech_recognition.text_publisher:main',
            'tts_subscriber_node = speech_recognition.text_subscriber:main',
            'audio_client_node = speech_recognition.audio_client_node:main', # <-- 쉼표(,) 추가
            'audio_server_node = speech_recognition.audio_server_node:main',
        ],
    },
)