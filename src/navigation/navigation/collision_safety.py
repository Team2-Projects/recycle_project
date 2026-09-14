"""ROS-independent collision gate, integrated from the Phase 2/3 bench tests.

Only the action adapter owns /cmd_vel.  This module never publishes a message.
Receipt times are monotonic.  Twist has no request identifier; matching checks
below reject incompatible responses, but do not constitute a DDS transaction ID.
"""
from dataclasses import dataclass, fields
import math
from typing import Optional

from navigation.tracking_control import Command


@dataclass(frozen=True)
class CollisionConfig:
    scan_timeout_sec: float = 0.40
    raw_timeout_sec: float = 0.50
    safe_timeout_sec: float = 0.50
    watchdog_period_sec: float = 0.05
    ready_timeout_sec: float = 3.0
    fault_timeout_sec: float = 3.0
    hold_timeout_sec: float = 3.0
    hold_clear_ratio: float = 0.80
    hold_clear_frames: int = 3
    clear_min_interval_sec: float = 0.04
    linear_zero_threshold: float = 0.002
    angular_zero_threshold: float = 0.005
    modification_tolerance: float = 1e-5
    footprint_front_x: float = 0.325
    footprint_rear_x: float = -0.196
    footprint_half_width: float = 0.14
    hard_stop_margin: float = 0.01
    geometry_timeout_sec: float = 1.0
    future_stamp_tolerance_sec: float = 0.10

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'collision_{field.name}: numeric value required')
            if not math.isfinite(value):
                raise ValueError(f'collision_{field.name}: finite value required')
            if field.name != 'footprint_rear_x' and value <= 0:
                raise ValueError(f'collision_{field.name} must be positive')
        if self.footprint_rear_x >= 0:
            raise ValueError('footprint_rear_x must be negative')
        if not isinstance(self.hold_clear_frames, int) or self.hold_clear_frames < 2:
            raise ValueError('hold_clear_frames must be an integer >= 2')
        if not 0 < self.hold_clear_ratio <= 1:
            raise ValueError('hold_clear_ratio must be in (0, 1]')
        if self.watchdog_period_sec >= min(self.scan_timeout_sec,
                                           self.raw_timeout_sec, self.safe_timeout_sec):
            raise ValueError('watchdog must run faster than freshness deadlines')
        if self.hold_timeout_sec <= (self.hold_clear_frames - 1) * self.clear_min_interval_sec:
            raise ValueError('hold timeout cannot accommodate the clear streak')

    def rectangle(self, margin=0.0):
        front = self.footprint_front_x + margin
        rear = self.footprint_rear_x - margin
        width = self.footprint_half_width + margin
        return ((front, -width), (front, width), (rear, width), (rear, -width))


@dataclass(frozen=True)
class SafetyDecision:
    command: Command = Command()
    state: str = 'IDLE'
    pause: bool = False
    realign: bool = False
    failure: str = ''


def finite_command(command):
    return math.isfinite(command.linear_x) and math.isfinite(command.angular_z)


def moving(command):
    return abs(command.linear_x) > 1e-9 or abs(command.angular_z) > 1e-9


def compatible(raw, safe, tolerance):
    """Humble STOP/APPROACH must return zero or a same-direction scalar reduction.

    Never cap individual components of an incompatible Twist: that could turn
    the collision-checked arc into a different, unchecked trajectory.
    """
    if not finite_command(raw) or not finite_command(safe):
        return False
    ratios = []
    for requested, returned in ((raw.linear_x, safe.linear_x),
                                (raw.angular_z, safe.angular_z)):
        if abs(requested) <= tolerance:
            if abs(returned) > tolerance:
                return False
        else:
            ratio = returned / requested
            if ratio < -tolerance or abs(returned) > abs(requested) + tolerance:
                return False
            ratios.append(ratio)
    if len(ratios) == 2:
        scale_tol = tolerance / max(abs(raw.linear_x), abs(raw.angular_z), tolerance)
        if abs(ratios[0] - ratios[1]) > max(1e-4, scale_tol * 2):
            return False
    return True


class CollisionSafety:
    """A single tracking action's gate and sticky collision state.

    - Continuous identical raw inputs do not reset safe freshness.
    - A new command / phase / stream restart requires a NEW matching safe sample.
    - HOLD's deadline survives zero probes, sensor gaps and boundary noise.
    - FINAL motion is never automatically resumed after an interruption.
    - A safety recovery only requests fresh visual REALIGN; it never resumes
      forward motion directly.
    """
    def __init__(self, config: CollisionConfig):
        self.config = config
        self.raw = Command()
        self.phase = 'IDLE'
        self.raw_at: Optional[float] = None
        self.safe_at: Optional[float] = None
        self.safe = Command()
        self.required_after: Optional[float] = None
        self.pending_at: Optional[float] = None
        self.safe_seq = 0
        self.processed_seq = 0
        self.hold_started_at: Optional[float] = None
        self.fault_started_at: Optional[float] = None
        self.recovering = False
        self.realign_required = False
        self.clear_count = 0
        self.clear_at: Optional[float] = None
        self.failure = ''
        self.final_armed = False
        self.state = 'IDLE'

    @property
    def suspended(self):
        return (self.hold_started_at is not None or self.recovering
                or self.realign_required)

    def request(self, command: Command, phase: str, now: float, force=False):
        if not finite_command(command) or not math.isfinite(now):
            self.fail('SAFETY_INVALID_COMMAND')
            return
        tol = self.config.modification_tolerance
        changed = (abs(command.linear_x - self.raw.linear_x) > tol
                   or abs(command.angular_z - self.raw.angular_z) > tol
                   or phase != self.phase)
        gap = self.raw_at is None or now - self.raw_at > self.config.raw_timeout_sec
        self.raw_at = now
        self.raw = command
        self.phase = phase
        if changed or gap or force:
            self.required_after = now
            # Repeated changing inputs cannot keep an unresponsive monitor alive.
            if self.pending_at is None:
                self.pending_at = now
            self.clear_count = 0
            self.clear_at = None

    def accept_safe(self, command: Command, now: float):
        if self.failure or self.raw_at is None or not math.isfinite(now):
            return False
        if not finite_command(command):
            self.fail('SAFETY_INVALID_OUTPUT')
            return False
        if not compatible(self.raw, command, self.config.modification_tolerance):
            # It may be a delayed reply to the previous request. Do not let it
            # renew the safe heartbeat or drive the new request.
            return False
        if self.required_after is not None and now < self.required_after:
            return False
        self.safe = command
        self.safe_at = now
        self.safe_seq += 1
        self.required_after = None
        self.pending_at = None
        return True

    def fail(self, reason):
        # First cause stays latched through subsequent watchdog failures.
        if not self.failure:
            self.failure = reason
        self.state = self.failure

    def acknowledge_realign(self, now):
        if self.failure or not self.realign_required:
            return False
        self.realign_required = False
        self.hold_started_at = None
        self.fault_started_at = None
        self.recovering = False
        self.clear_count = 0
        self.clear_at = None
        self.safe = Command()
        self.safe_at = None
        self.required_after = now
        self.pending_at = now
        self.final_armed = False
        self.state = 'REALIGN'
        return True

    def mark_final_started(self):
        if self.phase == 'FINAL_APPROACH' and not self.failure:
            self.final_armed = True

    def _zero_if_tiny(self, command):
        # Zero the WHOLE Twist only when all motion components are tiny. Keep
        # curvature for nonzero moving arcs rather than dropping angular.z alone.
        if (abs(command.linear_x) < self.config.linear_zero_threshold
                and abs(command.angular_z) < self.config.angular_zero_threshold):
            return Command()
        return command

    def _enough_clear(self, safe):
        if not moving(self.raw):
            return False  # A zero probe never proves that movement is safe.
        return all(
            abs(raw) <= self.config.modification_tolerance
            or (raw * val > 0 and abs(val) >= abs(raw) * self.config.hold_clear_ratio)
            for raw, val in ((self.raw.linear_x, safe.linear_x),
                             (self.raw.angular_z, safe.angular_z))
        )

    def _count_clear(self, now, clear):
        if self.safe_seq == self.processed_seq:
            return
        self.processed_seq = self.safe_seq
        if not clear:
            self.clear_count = 0
            self.clear_at = None
        elif self.clear_at is None:
            self.clear_count = 1
            self.clear_at = now
        elif now - self.clear_at >= self.config.clear_min_interval_sec:
            self.clear_count += 1
            self.clear_at = now
        # Bursts do not count as multiple independent clear observations.

    def evaluate(self, now, environment_ok=True, environment_reason=''):
        cfg = self.config
        if not math.isfinite(now):
            self.fail('SAFETY_CLOCK_ERROR')
        if self.failure:
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
        if (self.hold_started_at is not None
                and now - self.hold_started_at >= cfg.hold_timeout_sec):
            self.fail('COLLISION_BLOCKED')
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)

        reason = ''
        if not environment_ok:
            reason = environment_reason or 'SAFETY_NOT_READY'
        elif self.raw_at is None:
            reason = 'NO_RAW'
        elif not 0 <= now - self.raw_at <= cfg.raw_timeout_sec:
            reason = 'RAW_STALE'
        elif self.safe_at is None:
            reason = 'NO_SAFE'
        elif not 0 <= now - self.safe_at <= cfg.safe_timeout_sec:
            reason = 'SAFE_STALE'
        elif self.required_after is not None:
            reason = 'WAIT_FRESH_SAFE'

        if reason:
            expected_wait = (
                reason in ('NO_SAFE', 'WAIT_FRESH_SAFE')
                and self.pending_at is not None
                and now - self.pending_at <= cfg.safe_timeout_sec
            )
            if self.phase == 'FINAL_APPROACH' and self.final_armed:
                self.fail('FINAL_APPROACH_INTERRUPTED')
                return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            if not expected_wait:
                if self.fault_started_at is None:
                    self.fault_started_at = now
                self.recovering = True
                self.clear_count = 0
                self.clear_at = None
                if now - self.fault_started_at >= cfg.fault_timeout_sec:
                    self.fail('SAFETY_UNAVAILABLE')
                    return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            self.state = 'COLLISION_HOLD' if self.hold_started_at is not None else reason
            return SafetyDecision(state=self.state, pause=not expected_wait or self.suspended)

        safe = self._zero_if_tiny(self.safe)
        modified = (abs(self.raw.linear_x - safe.linear_x) > cfg.modification_tolerance
                    or abs(self.raw.angular_z - safe.angular_z) > cfg.modification_tolerance)
        raw_moving = moving(self.raw)
        if self.phase == 'FINAL_APPROACH' and raw_moving and modified:
            self.fail('FINAL_APPROACH_INTERRUPTED')
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
        if self.realign_required:
            return SafetyDecision(state='REALIGN_REQUIRED', pause=True, realign=True)

        if self.hold_started_at is not None or self.recovering:
            self._count_clear(
                now, self._enough_clear(safe)
                or (self.recovering and self.hold_started_at is None
                    and not raw_moving and not moving(safe)),
            )
            if self.clear_count >= cfg.hold_clear_frames:
                self.realign_required = True
                self.state = 'REALIGN_REQUIRED'
                return SafetyDecision(state=self.state, pause=True, realign=True)
            # A recovering stream with no motion probe cannot clear itself.
            # The adapter resumes visual alignment after a healthy zero-startup
            # handshake separately; normal active holds retain a motion probe.
            if (self.fault_started_at is not None
                    and now - self.fault_started_at >= cfg.fault_timeout_sec):
                self.fail('SAFETY_UNAVAILABLE')
                return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            self.state = ('COLLISION_HOLD' if self.hold_started_at is not None
                          else 'SAFETY_RECOVERY')
            return SafetyDecision(state=self.state, pause=True)

        if raw_moving and not moving(safe):
            self.hold_started_at = now
            self.clear_count = 0
            self.clear_at = None
            self.state = 'COLLISION_HOLD'
            return SafetyDecision(state=self.state, pause=True)
        self.state = 'COLLISION_SLOWING' if raw_moving and modified else self.phase
        return SafetyDecision(command=safe if raw_moving else Command(), state=self.state)
