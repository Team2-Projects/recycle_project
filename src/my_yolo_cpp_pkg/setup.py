from setuptools import setup
import os
from glob import glob

package_name = 'my_yolo_cpp_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 이 부분이 가장 중요합니다!
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hee',
    maintainer_email='hee@todo.todo',
    description='YOLO object detection node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_node = my_yolo_cpp_pkg.yolo_node:main',
            'brightness_node = my_yolo_cpp_pkg.brightness_analyzer:main',
            'norm_img= my_yolo_cpp_pkg.normalized_img:main',
            'collect_action_node = my_yolo_cpp_pkg.collect_action_node:main',
            'img_subs_node = my_yolo_cpp_pkg.img_subscribe:main',
            'img_save_node = my_yolo_cpp_pkg.img_save:main', 
            'return_object_id_node = my_yolo_cpp_pkg.detected_object_id:main'      
        ],                    
    },
)