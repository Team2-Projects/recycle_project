"""Opt-in DDS probe; private topics/domain only, no robot or production adapter.

Run separately from the in-process ROS doubles:
RECYCLE_ROS_PROBE=1 ROS_DOMAIN_ID=191 ROS_LOCALHOST_ONLY=1 python3 -m pytest -rx -s <this file>
An XFAIL explicitly records a reproduced, unresolved monitor-only scan gap.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest


@pytest.mark.skipif(os.environ.get('RECYCLE_ROS_PROBE') != '1', reason='opt-in isolated ROS2 probe')
def test_monitor_scan_loss_must_not_be_mistaken_for_clearance():
    assert os.environ.get('ROS_DOMAIN_ID') == '191'
    assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from ament_index_python.packages import get_package_prefix
    from geometry_msgs.msg import Point32, PolygonStamped, TransformStamped, Twist
    from lifecycle_msgs.srv import ChangeState
    from lifecycle_msgs.msg import Transition
    from sensor_msgs.msg import LaserScan
    from tf2_ros import StaticTransformBroadcaster

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from navigation.collision_safety import CollisionConfig, CollisionSafety, ScanInput, moving
    from navigation.tracking_control import Command

    params = Path(__file__).resolve().parents[1] / 'config/recycle_tracking.yaml'
    binary = Path(get_package_prefix('nav2_collision_monitor')) / 'lib/nav2_collision_monitor/collision_monitor'
    args = [str(binary), '--ros-args', '-r', '__node:=tracking_collision_monitor',
            '--params-file', str(params)]
    for name, value in {
        'cmd_vel_in_topic': '/monitor_probe/raw', 'cmd_vel_out_topic': '/monitor_probe/safe',
        'scan.topic': '/monitor_probe/scan',
        'HardStop.polygon_pub_topic': '/monitor_probe/hard_stop',
        'FootprintApproach.footprint_topic': '/monitor_probe/footprint',
        'FootprintApproach.polygon_pub_topic': '/monitor_probe/footprint_checked',
    }.items():
        args.extend(['-p', name + ':=' + value])
    rclpy.init()
    node = Node('monitor_scan_gap_probe')
    process = None
    log = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        scan_pub = node.create_publisher(LaserScan, '/monitor_probe/scan', qos_profile_sensor_data)
        raw_pub = node.create_publisher(Twist, '/monitor_probe/raw', 1)
        footprint_pub = node.create_publisher(PolygonStamped, '/monitor_probe/footprint', 1)
        broadcaster = StaticTransformBroadcaster(node)
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = 'base_footprint'
        transform.child_frame_id = 'base_scan'
        transform.transform.rotation.w = 1.
        broadcaster.sendTransform(transform)
        config = CollisionConfig()
        scan_input, safety = ScanInput(config), CollisionSafety(config)
        delivery = dict(enabled=True, obstacle=False, safe=None, decision=None, forwarded_at=None)

        def safe_callback(msg):
            now = time.monotonic()
            delivery['safe'] = msg.linear.x
            safety.accept_safe(Command(msg.linear.x, msg.angular.z), now)
            ready, reason = scan_input.health(now, node.get_clock().now().nanoseconds)
            delivery['decision'] = safety.evaluate(now, ready, reason, scan_input.sequence)

        node.create_subscription(Twist, '/monitor_probe/safe', safe_callback, 1)

        def tick():
            now = time.monotonic()
            scan = LaserScan()
            scan.header.stamp = node.get_clock().now().to_msg()
            scan.header.frame_id = 'base_scan'
            scan.angle_min, scan.angle_max, scan.angle_increment = -1., 1., .1
            scan.range_min, scan.range_max = .05, 10.
            scan.ranges = [.20 if delivery['obstacle'] else 2.] * 21
            scan_input.receive(scan, now, node.get_clock().now().nanoseconds)
            if delivery['enabled']:
                scan_pub.publish(scan)
                delivery['forwarded_at'] = now
            footprint = PolygonStamped()
            footprint.header.stamp = scan.header.stamp
            footprint.header.frame_id = 'base_footprint'
            footprint.polygon.points = [Point32(x=x, y=y, z=0.) for x, y in config.rectangle()]
            footprint_pub.publish(footprint)
            safety.request(Command(.08, 0.), 'APPROACH', now)
            raw = Twist()
            raw.linear.x = .08
            raw_pub.publish(raw)

        node.create_timer(.05, tick)
        lifecycle = node.create_client(ChangeState, '/tracking_collision_monitor/change_state')
        assert lifecycle.wait_for_service(timeout_sec=8.), 'Monitor lifecycle unavailable'
        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            request = ChangeState.Request()
            request.transition.id = transition
            future = lifecycle.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=8.)
            assert future.done() and future.result().success

        def spin_for(duration):
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                assert process.poll() is None, 'Monitor exited unexpectedly'
                rclpy.spin_once(node, timeout_sec=.05)

        spin_for(2.)
        assert delivery['decision'] is not None and moving(delivery['decision'].command)
        delivery['obstacle'] = True
        spin_for(.6)
        assert delivery['safe'] == 0., 'Healthy monitor did not stop for the test obstacle'
        delivery['obstacle'] = False
        spin_for(.6)
        assert delivery['safe'] > 0., 'Monitor did not recover after the obstacle cleared'
        # Begin a fresh guard attempt after old zero replies have drained.
        safety = CollisionSafety(config)
        safety.request(Command(.08, 0.), 'APPROACH', time.monotonic())
        spin_for(1.2)
        assert moving(delivery['decision'].command)
        delivery['enabled'] = False
        delivery['obstacle'] = True  # only the guard now receives the obstacle scan
        spin_for(1.2)
        assert time.monotonic() - delivery['forwarded_at'] > config.scan_timeout_sec
        assert scan_input.health(time.monotonic(), node.get_clock().now().nanoseconds)[0]
        log.seek(0)
        assert b'Ignoring the source' in log.read(), 'Monitor timeout was not reproduced'
        if moving(delivery['decision'].command):
            pytest.xfail('Humble 1.1.20: monitor scan timed out, but fresh raw/safe and guard scan still permit motion')
        assert delivery['safe'] == 0.
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.)
        node.destroy_node()
        rclpy.shutdown()
        log.close()
