from setuptools import setup

package_name = 'my_yolo_cpp_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
            'img_subs_node = my_yolo_cpp_pkg.img_subscribe:main'       
        ],                    
    },
)