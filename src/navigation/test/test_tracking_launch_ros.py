"""Actual launch, lifecycle manager, Monitor and Tracking Action on isolated DDS.

Only perception/service inputs are synthetic; no physical robot is connected.
"""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import pytest
import yaml

pytestmark = pytest.mark.skipif(os.environ.get('RECYCLE_ROS_PROBE') != '1',
                               reason='opt-in isolated ROS2 launch probe')


def test_existing_launch_and_real_tracking_action_release(tmp_path):
    assert os.environ.get('ROS_DOMAIN_ID') == '191'
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist, TransformStamped
    from sensor_msgs.msg import LaserScan
    from tf2_ros import StaticTransformBroadcaster
    from lifecycle_msgs.srv import GetState
    from my_yolo_msgs.msg import DetectedObject
    from my_yolo_msgs.srv import SetTracking
    from navigation_interface.action import RecycleActionMsg
    from navigation_interface.msg import CollisionCommand

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/recycle_tracking.yaml').read_text())
    # Test near-stop mode only; all collision geometry and timing are production values.
    config['recycle_tracking_node']['ros__parameters']['stop_only_test_mode'] = True
    params = tmp_path / 'tracking.yaml'
    params.write_text(yaml.safe_dump(config))
    rclpy.init()
    node = Node('tracking_launch_probe')
    process, log = None, tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(['ros2', 'launch', 'navigation', 'navigation.launch.py',
            'tracking_only:=true', 'tracking_params:=' + str(params)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        output, raw, modes = [], [], []
        node.create_subscription(Twist, '/cmd_vel', output.append, 10)
        node.create_subscription(CollisionCommand, '/tracking_cmd_vel_raw', raw.append, 10)
        scan_pub = node.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        detection_pub = node.create_publisher(DetectedObject, '/classified_detected_object_info', 1)
        tf = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = 'base_footprint', 'base_scan'
        transform.transform.rotation.w = 1.
        tf.sendTransform(transform)
        active, scan_on, started, last_detection = False, False, 0., 0.
        def mode(request, response):
            nonlocal active, started
            active, started = request.enable, time.monotonic()
            modes.append(request.enable)
            response.success, response.reason = True, 'OK'
            return response
        node.create_service(SetTracking, 'set_tracking_mode', mode)
        def sensors():
            nonlocal last_detection
            now = time.monotonic()
            if scan_on:
                scan = LaserScan()
                scan.header.frame_id = 'base_scan'
                scan.header.stamp = node.get_clock().now().to_msg()
                scan.angle_min, scan.angle_max, scan.angle_increment = -1., 1., .1
                scan.range_min, scan.range_max, scan.ranges = .05, 10., [5.] * 21
                scan_pub.publish(scan)
            if active and now - last_detection >= .19:
                last_detection = now
                bottom = min(435., 280. + max(0., now - started - 1.) * 45.)
                msg = DetectedObject(id=0, confidence=.9, coord=[350., bottom - 25., 40., 50.])
                detection_pub.publish(msg)
        node.create_timer(.05, sensors)
        def spin(duration, done=lambda: False):
            until = time.monotonic() + duration
            while time.monotonic() < until and not done():
                assert process.poll() is None
                rclpy.spin_once(node, timeout_sec=.05)
        client = ActionClient(node, RecycleActionMsg, 'recycle_tracking_action')
        assert client.wait_for_server(timeout_sec=8.)
        state_client = node.create_client(GetState, '/tracking_collision_monitor/get_state')
        assert state_client.wait_for_service(timeout_sec=5.)
        active_monitor = False
        until = time.monotonic() + 8.
        while time.monotonic() < until:
            future = state_client.call_async(GetState.Request())
            spin(1., future.done)
            if future.done() and future.result().current_state.id == 3:
                active_monitor = True
                break
        assert active_monitor
        spin(.5)
        assert not raw and not output  # No idle handshake, no idle wheel heartbeat.
        def attempt():
            future = client.send_goal_async(RecycleActionMsg.Goal(index=0))
            spin(3., future.done)
            assert future.done() and future.result().accepted
            result_future = future.result().get_result_async()
            spin(15., result_future.done)
            assert result_future.done()
            return result_future.result()
        missing_scan = attempt()
        assert missing_scan.status == 6
        assert missing_scan.result.message.startswith('SAFETY_NOT_READY:')
        assert all(m.linear.x == 0 and m.angular.z == 0 for m in output)
        scan_on = True
        spin(.6)
        healthy = attempt()
        assert healthy.result.message.startswith('TEST_STOP:'), healthy.result.message
        assert any(m.linear.x > 0 for m in output)
        assert modes == [True, False, True, False]
        spin(.3)  # Drain already queued zero commands after the terminal result.
        count = len(output), len(raw)
        spin(.8)
        assert (len(output), len(raw)) == count
        assert output[-1].linear.x == output[-1].angular.z == 0
    finally:
        if process is not None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.)
        log.seek(0)
        print(log.read().decode(errors='replace'))
        log.close()
        node.destroy_node()
        rclpy.shutdown()
