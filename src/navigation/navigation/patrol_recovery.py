"""Small ROS-independent return-to-patrol policy (not a motion controller).

A failed collection and a failed navigation stack are different failures.
This module never publishes velocities or sends goals. AutoNav performs the
physical handoff only after its live inputs and asynchronous services are ready.
"""
from dataclasses import dataclass, fields
import math


# A final-motion interruption retires ONE attempt, not the patrol mission.
# Unknown/programming errors, explicit cancellation and TEST_STOP remain manual.
# This list authorizes readiness checks, never unconditionally authorizes motion.
AUTO_PATROL_REASONS = frozenset({
    'COLLISION_BLOCKED', 'LOST_TARGET', 'ALIGN_TIMEOUT', 'APPROACH_TIMEOUT',
    'SAFETY_NOT_READY', 'SAFETY_UNAVAILABLE', 'SENSOR_STALE', 'VISION_NOT_READY',
    'TRACKING_GOAL_REJECTED', 'FINAL_APPROACH_INTERRUPTED',
})


@dataclass(frozen=True)
class PatrolRecoveryConfig:
    enabled: bool = True
    final_interrupted_enabled: bool = True
    period_sec: float = 0.20
    odom_timeout_sec: float = 0.60
    future_stamp_tolerance_sec: float = 0.10
    stopped_linear_speed: float = 0.01
    stopped_angular_speed: float = 0.02
    service_timeout_sec: float = 2.0
    service_retry_sec: float = 3.0
    release_distance_m: float = 0.10

    def __post_init__(self):
        for name in ('enabled', 'final_interrupted_enabled'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'patrol_recovery_{name} must be boolean')
        for field in fields(self):
            if field.name in ('enabled', 'final_interrupted_enabled'):
                continue
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'patrol_recovery_{field.name} must be finite and positive')
        if self.period_sec >= self.odom_timeout_sec:
            raise ValueError('recovery poll must be faster than receipt deadlines')


class OdomEvidence:
    """Bounded last-good pose/speed evidence in odom coordinates, never map jumps."""
    def __init__(self, config: PatrolRecoveryConfig):
        self.config = config
        self.at = None
        self.stamp_ns = None
        self.x = self.y = self.linear = self.angular = 0.0

    def receive(self, now, ros_ns, stamp_ns, frame, x, y, linear, angular):
        values = (now, x, y, linear, angular)
        if (type(stamp_ns) is not int or stamp_ns <= 0 or frame != 'odom'
                or not all(math.isfinite(v) for v in values)):
            return False
        stamp_age = (ros_ns - stamp_ns) * 1e-9
        if not -self.config.future_stamp_tolerance_sec <= stamp_age <= self.config.odom_timeout_sec:
            return False
        if self.at is not None and now < self.at:
            return False
        if self.stamp_ns is not None and stamp_ns <= self.stamp_ns:
            if self.fresh(now, ros_ns):
                return False
            # Permit a fresh new timestamp epoch after an actual outage.
            if stamp_ns == self.stamp_ns:
                return False
        self.at, self.stamp_ns = now, stamp_ns
        self.x, self.y, self.linear, self.angular = x, y, linear, angular
        return True

    def fresh(self, now, ros_ns):
        return (self.at is not None and 0 <= now - self.at <= self.config.odom_timeout_sec
                and -self.config.future_stamp_tolerance_sec
                <= (ros_ns - self.stamp_ns) * 1e-9 <= self.config.odom_timeout_sec)

    def stopped(self, now, ros_ns):
        return (self.fresh(now, ros_ns)
                and abs(self.linear) <= self.config.stopped_linear_speed
                and abs(self.angular) <= self.config.stopped_angular_speed)


class AcquisitionLock:
    """Suppress *all* new collection until real patrol movement + goal success.

    The detector has no persistent instance ID. Do not pretend a class label is
    a per-object blacklist. Cancels/rejections/instant zero-distance arrivals do
    not release this lock. Odom resets must not masquerade as movement.
    """
    def __init__(self, distance=0.10, max_gap_sec=0.60):
        self.distance = distance
        self.max_gap_sec = max_gap_sec
        self.locked = False
        self.origin = self.last_pose = None
        self.last_at = None
        self.displacement = 0.0

    def engage(self):
        self.locked = True
        self.origin = self.last_pose = None
        self.last_at = None
        self.displacement = 0.0

    def begin_patrol(self, x, y, now):
        if not self.locked:
            return
        self.origin = self.last_pose = (x, y)
        self.last_at = now
        self.displacement = 0.0

    def observe(self, x, y, now):
        if not self.locked or self.origin is None:
            return
        dt = now - self.last_at
        step = math.hypot(x - self.last_pose[0], y - self.last_pose[1])
        if dt <= 0:
            return
        if dt > self.max_gap_sec or step > max(0.05, dt * 1.0):
            # Lose progress on a jump/dropout rather than unlocking from it.
            self.begin_patrol(x, y, now)
            return
        self.last_at, self.last_pose = now, (x, y)
        self.displacement = math.hypot(x - self.origin[0], y - self.origin[1])

    def waypoint_succeeded(self):
        if self.locked and self.origin is not None and self.displacement >= self.distance:
            self.locked = False
            return True
        return False


@dataclass
class ReturnPlan:
    reason: str
    x: float
    y: float
    created_at: float
    servo_confirmed: bool = False
    manual: bool = False
    destination: str = 'PATROL'
