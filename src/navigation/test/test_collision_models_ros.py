"""Opt-in comparisons against the installed Humble executable, on private DDS.

Run separately from the in-process ROS doubles, with the same environment as
test_monitor_scan_gap_ros.py. The fixed model is a candidate, never production.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('RECYCLE_ROS_PROBE') != '1', reason='opt-in isolated ROS2 comparison')


@pytest.fixture
def monitors():
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
    from navigation_interface.msg import CollisionCommand
    import yaml

    params = Path(__file__).resolve().parents[1] / 'config/recycle_tracking.yaml'
    legacy_binary = Path(get_package_prefix('nav2_collision_monitor')) / 'lib/nav2_collision_monitor/collision_monitor'
    checked_binary = Path(get_package_prefix('navigation')) / 'lib/navigation/tracking_collision_monitor'
    points = yaml.safe_load(params.read_text())['tracking_collision_monitor']['ros__parameters']['FootprintApproach']['points']
    rclpy.init()

    @contextmanager
    def run(model='checked', sim_time=False):
        node = Node('collision_model_probe')
        log = tempfile.TemporaryFile()
        process = None
        try:
            checked = model == 'checked'
            packet_type = CollisionCommand if checked else Twist
            result = NS(last=None, stamp=None, sent=0, guard_receipts=0,
                        sim_clock=1000., clock_paused=False)
            overrides = {
                'cmd_vel_in_topic': '/model_probe/raw', 'cmd_vel_out_topic': '/model_probe/safe',
                'scan.topic': '/model_probe/scan',
                'HardStop.polygon_pub_topic': '/model_probe/stop',
                'FootprintApproach.footprint_topic': '/model_probe/footprint',
                'FootprintApproach.polygon_pub_topic': '/model_probe/checked',
            }
            if checked:
                overrides['FootprintApproach.footprint_topic'] = "''"
            if sim_time:
                from rclpy.parameter import Parameter
                from rosgraph_msgs.msg import Clock
                overrides['use_sim_time'] = 'true'
                node.set_parameters([Parameter('use_sim_time', value=True)])
                clock_pub = node.create_publisher(Clock, '/clock', 10)
            if not checked:
                overrides['stop_pub_timeout'] = '1000000000.0'
            if model == 'fixed':
                # Illustrative outer rectangle: footprint + 0.16 m on all
                # sides (0.08 m/s * 2 s), NOT a calibrated braking distance.
                overrides.update({
                    'polygons': '[HardStop, OuterSlow]', 'OuterSlow.type': 'polygon',
                    'OuterSlow.action_type': 'slowdown', 'OuterSlow.max_points': '3',
                    'OuterSlow.points': '[0.485, 0.30, 0.485, -0.30, -0.356, -0.30, -0.356, 0.30]',
                    'OuterSlow.slowdown_ratio': '0.5', 'OuterSlow.visualize': 'false',
                })
            args = [str(checked_binary if checked else legacy_binary), '--ros-args', '-r', '__node:=tracking_collision_monitor',
                    '--params-file', str(params)]
            for name, value in overrides.items():
                args.extend(['-p', name + ':=' + value])
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            raw_pub = node.create_publisher(packet_type, '/model_probe/raw', 1)
            scan_pub = node.create_publisher(LaserScan, '/model_probe/scan', qos_profile_sensor_data)
            source_pub = node.create_publisher(LaserScan, '/model_probe/source_scan', qos_profile_sensor_data)
            def guard_receive(msg):
                result.guard_receipts += 1
            node.create_subscription(LaserScan, '/model_probe/source_scan', guard_receive, qos_profile_sensor_data)
            footprint_pub = node.create_publisher(PolygonStamped, '/model_probe/footprint', 1)
            broadcaster = StaticTransformBroadcaster(node)
            transform = TransformStamped()
            transform.header.stamp = node.get_clock().now().to_msg()
            transform.header.frame_id = 'base_footprint'
            transform.child_frame_id = 'base_scan'
            transform.transform.rotation.w = 1.
            broadcaster.sendTransform(transform)
            replies = []
            def receive(msg):
                replies.append(msg)
                result.last = msg
            node.create_subscription(packet_type, '/model_probe/safe', receive, 1)

            lifecycle = node.create_client(ChangeState, '/tracking_collision_monitor/change_state')
            assert lifecycle.wait_for_service(timeout_sec=8.)
            def change_state(transition):
                request = ChangeState.Request()
                request.transition.id = transition
                future = lifecycle.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=8.)
                assert future.done() and future.result().success
            for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
                change_state(transition)

            def sample(command, segments=(), duration=.55, stamp_offset=0., scan_on=True,
                       fixed_stamp=None, frame='base_scan', malformed=False, scan_mutator=None):
                # Synthetic 2D lidar, axis-aligned finite wall segments.
                scan = LaserScan()
                scan.header.frame_id = frame
                scan.angle_min = -math.pi
                scan.angle_increment = 2. * math.pi / 1440
                scan.angle_max = scan.angle_min + 1439 * scan.angle_increment
                scan.range_min, scan.range_max = .05, 10.
                ranges = []
                for i in range(1440):
                    angle = scan.angle_min + i * scan.angle_increment
                    dx, dy = math.cos(angle), math.sin(angle)
                    distance = 10.
                    for x1, y1, x2, y2 in segments:
                        if x1 == x2 and abs(dx) > 1e-9:
                            d = x1 / dx
                            inside = min(y1, y2) <= d * dy <= max(y1, y2)
                        elif y1 == y2 and abs(dy) > 1e-9:
                            d = y1 / dy
                            inside = min(x1, x2) <= d * dx <= max(x1, x2)
                        else:
                            continue
                        if inside and scan.range_min <= d < distance:
                            distance = d
                    ranges.append(distance)
                scan.ranges = ranges
                raw = Twist()
                raw.linear.x, raw.angular.z = command
                footprint = PolygonStamped()
                footprint.header.frame_id = 'base_footprint'
                footprint.polygon.points = [Point32(x=x, y=y, z=0.) for x, y in zip(points[::2], points[1::2])]
                start = time.monotonic()
                replies.clear()
                while time.monotonic() - start < duration:
                    assert process.poll() is None, 'Monitor exited unexpectedly'
                    if sim_time:
                        if not result.clock_paused:
                            result.sim_clock += .04
                        clock = Clock()
                        clock.clock.sec, clock.clock.nanosec = divmod(round(result.sim_clock * 1e9), 10**9)
                        clock_pub.publish(clock)
                        rclpy.spin_once(node, timeout_sec=.01)
                    ros_ns = node.get_clock().now().nanoseconds
                    ns = ros_ns + round(stamp_offset * 1e9) if fixed_stamp is None else fixed_stamp
                    result.stamp = ns
                    scan.header.stamp.sec, scan.header.stamp.nanosec = divmod(ns, 10**9)
                    if malformed:
                        scan.ranges = []
                    if scan_mutator is not None:
                        scan_mutator(scan)
                    source_pub.publish(scan)
                    if scan_on:
                        scan_pub.publish(scan)
                    footprint.header.stamp = node.get_clock().now().to_msg()
                    footprint_pub.publish(footprint)
                    result.sent += 1
                    raw_pub.publish(CollisionCommand(request_id=result.sent, velocity=raw) if checked else raw)
                    until = time.monotonic() + .04
                    while time.monotonic() < until:
                        rclpy.spin_once(node, timeout_sec=.01)
                assert len(replies) >= 2, (
                    'Missing monitor output; subscriptions=' + repr(
                        node.get_subscriber_names_and_types_by_node('tracking_collision_monitor', '/')))
                velocity = replies[-1].velocity if checked else replies[-1]
                if checked:
                    assert replies[-1].request_id <= result.sent
                return velocity.linear.x, velocity.angular.z

            sample((0., 0.), duration=1.)  # Discovery, footprint and TF warmup.
            result.sample, result.node, result.checked = sample, node, checked
            result.change_state, result.replies, result.raw_pub = change_state, replies, raw_pub
            yield result
        finally:
            if sys.exc_info()[0] is not None:
                log.seek(0)
                print(log.read().decode(errors='replace'))
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.)
            node.destroy_node()
            log.close()

    yield run
    rclpy.shutdown()


def test_static_approach_matches_installed_humble_geometry(monitors):
    scenes = {
        'clear': ((.08, 0.), ()),
        'front_wall': ((.08, 0.), ((.42, -.4, .42, .4),)),
        'hard_stop': ((.08, 0.), ((.32, -.4, .32, .4),)),
        'parallel_side': ((.08, 0.), ((-1., .18, 1., .18),)),
        'left_tip': ((0., .12), ((.30, .165, .33, .165),)),
        'right_tip': ((0., -.12), ((.30, -.165, .33, -.165),)),
        'forward_turn': ((.08, .06), ((.30, .165, .33, .165),)),
        'away_from_wall': ((-.08, 0.), ((.42, -.4, .42, .4),)),
    }
    results = {}
    for model in ('approach', 'checked', 'fixed'):
        with monitors(model) as probe:
            results[model] = {name: probe.sample(command, walls)
                              for name, (command, walls) in scenes.items()}
    print('MODEL_COMPARISON=' + json.dumps(results, sort_keys=True))
    for name in scenes:
        assert results['checked'][name] == pytest.approx(results['approach'][name])
    assert results['checked']['hard_stop'] == (0., 0.)
    assert results['checked']['parallel_side'] == (.08, 0.)
    assert results['fixed']['parallel_side'] == (.04, 0.)
    assert results['checked']['left_tip'][1] < results['fixed']['left_tip'][1]


def test_direct_scan_retains_last_good_obstacle(monitors):
    with monitors() as probe:
        assert probe.sample((.08, 0.), ((.32, -.4, .32, .4),)) == (0., 0.)
        # No relay/Guard: Monitor rejects a late scan without losing the wall.
        assert probe.sample((.08, 0.), duration=.15, stamp_offset=-2.) == (0., 0.)
        assert probe.last.input_valid
        assert probe.sample((.08, 0.), duration=.55, stamp_offset=-2.) == (0., 0.)
        assert not probe.last.input_valid


def test_humble_cli_overrides_private_topics(monitors):
    with monitors() as probe:
        assert probe.sample((.08, 0.)) == (.08, 0.)
        assert probe.last.input_valid
        assert probe.last.request_id == probe.sent
        names = dict(probe.node.get_subscriber_names_and_types_by_node('tracking_collision_monitor', '/'))
        assert '/model_probe/raw' in names and '/model_probe/scan' in names
        assert '/scan' not in names and '/tracking_cmd_vel_raw' not in names


def test_checked_monitor_missing_tf_is_not_clear_space(monitors):
    with monitors() as probe:
        assert probe.sample((.08, 0.), frame='missing_sensor_tf', duration=1.) == (0., 0.)
        assert not probe.last.input_valid
        assert probe.sample((.08, 0.), duration=1.) == (.08, 0.)
        assert probe.last.input_valid
