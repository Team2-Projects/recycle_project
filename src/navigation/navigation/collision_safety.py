"""ROS-independent collision gate for tracking commands.

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
    # Health recovery is distinct from collision-clear evidence.
    recovery_stable_sec: float = 0.50
    recovery_frames: int = 3
    scan_restart_frames: int = 3
    auto_recovery_limit: int = 2

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'collision_{field.name}: numeric value required')
            if not math.isfinite(value):
                raise ValueError(f'collision_{field.name}: finite value required')
            if field.name != 'footprint_rear_x' and value <= 0:
                raise ValueError(f'collision_{field.name} must be positive')
        if type(self.auto_recovery_limit) is not int or self.auto_recovery_limit < 1:
            raise ValueError('auto_recovery_limit must be an integer >= 1')
        if self.footprint_rear_x >= 0:
            raise ValueError('footprint_rear_x must be negative')
        if not isinstance(self.hold_clear_frames, int) or self.hold_clear_frames < 2:
            raise ValueError('hold_clear_frames must be an integer >= 2')
        for name in ('recovery_frames', 'scan_restart_frames'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        if self.recovery_stable_sec >= min(self.ready_timeout_sec, self.fault_timeout_sec):
            raise ValueError('recovery confirmation must fit inside ready/fault budgets')
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


class InputRecovery:
    """Count distinct scan/safe receipts, not repeated calls to evaluate().

    Used both by startup readiness and recovery after a transient input fault.
    Receipt times and dwell are measured on one local monotonic clock.
    """
    def __init__(self, config: CollisionConfig):
        self.config = config
        self.reset()

    def reset(self, scan_sequence=None, safe_sequence=None):
        self.started_at = None
        self.scan_sequence = scan_sequence
        self.safe_sequence = safe_sequence
        self.scan_count = self.safe_count = 0
        self.scan_at = self.safe_at = None

    def push(self, now, scan_sequence, safe_sequence):
        if not math.isfinite(now):
            return False
        if self.started_at is None:
            self.started_at = now
        for kind, sequence in (('scan', scan_sequence), ('safe', safe_sequence)):
            if sequence is None or sequence == getattr(self, kind + '_sequence'):
                continue
            setattr(self, kind + '_sequence', sequence)
            previous_at = getattr(self, kind + '_at')
            if (previous_at is None
                    or now - previous_at >= self.config.clear_min_interval_sec):
                setattr(self, kind + '_count', getattr(self, kind + '_count') + 1)
                setattr(self, kind + '_at', now)
        return (now - self.started_at >= self.config.recovery_stable_sec
                and self.scan_count >= self.config.recovery_frames
                and self.safe_count >= self.config.recovery_frames)


class ScanInput:
    """Last GOOD scan metadata, without changing the original acquisition stamp.

    The adapter forwards ONLY accepted scans to the private monitor input.
    This is essential: ignoring a duplicate/malformed scan in the guard alone
    would leave Humble Collision Monitor consuming a different (bad) sample.
    A discarded sample never renews receipt freshness. Persistent clock skew or
    backlog is NOT fixed by stamping old data with the local current time.
    """
    def __init__(self, config: CollisionConfig):
        self.config = config
        self.received_at: Optional[float] = None
        self.last_message_at: Optional[float] = None
        self.stamp_ns: Optional[int] = None
        self.frame = ''
        self.sequence = 0
        self.last_rejection = ''
        self.rejected_count = 0
        self.last_offered_age_sec: Optional[float] = None
        self.restart_count = 0
        self._restart_stamp = None
        self._restart_at = None
        self._restart_frame = ''

    def _clear_restart(self):
        self.restart_count = 0
        self._restart_stamp = self._restart_at = None
        self._restart_frame = ''

    def reject(self, reason, now=None):
        if now is not None and math.isfinite(now):
            self.last_message_at = now
        self.last_rejection = reason
        self.rejected_count += 1
        return False

    def receive(self, msg, now, ros_now_ns):
        """Validate the same original LaserScan for tracking and Nav2 handoff."""
        try:
            stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)
            frame = msg.header.frame_id
            numeric = (msg.angle_min, msg.angle_max, msg.angle_increment,
                       msg.range_min, msg.range_max)
            valid = (isinstance(frame, str) and bool(frame) and stamp_ns > 0
                     and 0 <= int(msg.header.stamp.nanosec) < 1000000000
                     and len(msg.ranges) >= 2
                     and all(math.isfinite(v) for v in numeric)
                     and msg.angle_increment > 0 and msg.range_max > msg.range_min
                     and any(math.isfinite(v) and max(msg.range_min, 0.001) <= v <= msg.range_max
                             for v in msg.ranges))
            if valid:
                return self.offer(stamp_ns, frame, now, ros_now_ns)
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass
        return self.reject('SCAN_INVALID', now)

    def health(self, now, ros_now_ns):
        if self.received_at is None:
            return False, self.last_rejection or 'NO_SCAN'
        age = now - self.received_at
        if not math.isfinite(age) or age < 0:
            return False, 'SCAN_RECEIPT_CLOCK_ORDER'
        if age > self.config.scan_timeout_sec:
            # Keep receipt timeout distinct from clock/observation age faults.
            return False, 'SCAN_STALE'
        stamp_age = (ros_now_ns - self.stamp_ns) * 1e-9
        if stamp_age < -self.config.future_stamp_tolerance_sec:
            return False, 'SCAN_STAMP_FUTURE'
        if stamp_age > self.config.scan_timeout_sec:
            return False, 'SCAN_STAMP_OLD'
        return True, ''

    def offer(self, stamp_ns, frame, now, ros_now_ns):
        if math.isfinite(now):
            self.last_message_at = now
        if (isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int)
                or stamp_ns <= 0 or not isinstance(frame, str) or not frame
                or not math.isfinite(now)):
            return self.reject('SCAN_INVALID')
        if self.received_at is not None and now < self.received_at:
            return self.reject('SCAN_RECEIPT_CLOCK_ORDER')
        self.last_offered_age_sec = (ros_now_ns - stamp_ns) * 1e-9
        if self.last_offered_age_sec < -self.config.future_stamp_tolerance_sec:
            self._clear_restart()
            return self.reject('SCAN_STAMP_FUTURE')
        if self.last_offered_age_sec > self.config.scan_timeout_sec:
            self._clear_restart()
            return self.reject('SCAN_STAMP_OLD')

        # Ignore duplicates/late frames while last-good data is still valid.
        # They are not evidence of a fresh scan nor an immediate global failure.
        reordering = (self.stamp_ns is not None
                      and (stamp_ns <= self.stamp_ns or frame != self.frame))
        if reordering:
            healthy, _ = self.health(now, ros_now_ns)
            if healthy:
                return self.reject('SCAN_DUPLICATE' if stamp_ns == self.stamp_ns
                                   and frame == self.frame else 'SCAN_OUT_OF_ORDER_OR_FRAME_CHANGE')
            # A sensor or clock reset must not permanently wait for the old
            # high-water mark. Re-establish a new epoch while output is stopped,
            # with recent original stamps increasing over several real samples.
            # Repeating one old sample can never satisfy this condition.
            if self._restart_stamp is not None and frame == self._restart_frame:
                if stamp_ns == self._restart_stamp:
                    return self.reject('SCAN_DUPLICATE')
                gap = now - self._restart_at
                if (stamp_ns > self._restart_stamp
                        and self.config.clear_min_interval_sec <= gap <= self.config.scan_timeout_sec):
                    self.restart_count += 1
                elif stamp_ns > self._restart_stamp and 0 <= gap < self.config.clear_min_interval_sec:
                    return self.reject('SCAN_RESTART_BURST')
                else:
                    self.restart_count = 1
            else:
                self.restart_count = 1
            self._restart_stamp, self._restart_at = stamp_ns, now
            self._restart_frame = frame
            if self.restart_count < self.config.scan_restart_frames:
                return self.reject('SCAN_STREAM_RECONFIRMING')

        self.received_at, self.stamp_ns, self.frame = now, stamp_ns, frame
        self.sequence += 1
        self.last_rejection = ''
        self._clear_restart()
        return True


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
        self.last_fault_reason = ''
        self.recovery = InputRecovery(config)

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
        self.recovery.reset()
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

    def evaluate(self, now, environment_ok=True, environment_reason='', scan_sequence=None):
        cfg = self.config
        if not math.isfinite(now):
            self.fail('SAFETY_CLOCK_ERROR')
        if self.failure:
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
        # Diagnose missing safety inputs before deciding that a wall is still
        # present. An expired scan/safe result is not fresh collision evidence.
        reason = ''
        if not environment_ok:
            reason = environment_reason or 'SAFETY_NOT_READY'
        elif self.raw_at is None:
            reason = 'NO_RAW'
        elif now < self.raw_at:
            reason = 'RAW_RECEIPT_CLOCK_ORDER'
        elif now - self.raw_at > cfg.raw_timeout_sec:
            reason = 'RAW_STALE'
        elif self.safe_at is None:
            reason = 'NO_SAFE'
        elif now < self.safe_at:
            reason = 'SAFE_RECEIPT_CLOCK_ORDER'
        elif now - self.safe_at > cfg.safe_timeout_sec:
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
                self.last_fault_reason = reason
                self.fail('FINAL_APPROACH_INTERRUPTED')
                return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            if self.recovering or not expected_wait:
                self.recovery.reset(scan_sequence, self.safe_seq)
            if not expected_wait:
                self.last_fault_reason = reason
                if self.fault_started_at is None:
                    self.fault_started_at = now
                self.recovering = True
                self.clear_count = 0
                self.clear_at = None
                if now - self.fault_started_at >= cfg.fault_timeout_sec:
                    self.fail('SAFETY_UNAVAILABLE')
                    return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            self.state = reason
            return SafetyDecision(state=self.state, pause=not expected_wait or self.suspended)

        safe = self._zero_if_tiny(self.safe)
        modified = (abs(self.raw.linear_x - safe.linear_x) > cfg.modification_tolerance
                    or abs(self.raw.angular_z - safe.angular_z) > cfg.modification_tolerance)
        raw_moving = moving(self.raw)
        if self.phase == 'FINAL_APPROACH' and raw_moving and modified:
            self.fail('FINAL_APPROACH_INTERRUPTED')
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
        if self.recovering:
            # Prove that inputs, rather than merely control ticks, recovered.
            # A working monitor returning a reduced speed is NOT a sensor fault.
            recovered = self.recovery.push(now, scan_sequence, self.safe_seq)
            if self.hold_started_at is not None:
                self._count_clear(now, self._enough_clear(safe))
            if (self.fault_started_at is not None
                    and now - self.fault_started_at >= cfg.fault_timeout_sec):
                self.fail('SAFETY_UNAVAILABLE')
                return SafetyDecision(state=self.failure, pause=True, failure=self.failure)
            if not recovered:
                self.state = 'SAFETY_RECOVERY'
                return SafetyDecision(state=self.state, pause=True)
            self.recovering = False
            self.fault_started_at = None
            if self.hold_started_at is not None and self.clear_count >= cfg.hold_clear_frames:
                # Fresh evidence confirms that the obstacle disappeared during
                # the input gap. Do not abort based only on an old hold timer.
                self.realign_required = True
            elif raw_moving and not moving(safe):
                # Communication is healthy again but an obstacle still blocks
                # the probe. Diagnose collision, not a permanent input outage.
                if self.hold_started_at is None:
                    self.hold_started_at = now
            elif self.hold_started_at is None:
                self.realign_required = True

        if self.realign_required:
            self.state = 'REALIGN_REQUIRED'
            return SafetyDecision(state=self.state, pause=True, realign=True)
        if (self.hold_started_at is not None
                and now - self.hold_started_at >= cfg.hold_timeout_sec):
            self.fail('COLLISION_BLOCKED')
            return SafetyDecision(state=self.failure, pause=True, failure=self.failure)

        if self.hold_started_at is not None:
            self._count_clear(
                now, self._enough_clear(safe),
            )
            if self.clear_count >= cfg.hold_clear_frames:
                self.realign_required = True
                self.state = 'REALIGN_REQUIRED'
                return SafetyDecision(state=self.state, pause=True, realign=True)
            self.state = 'COLLISION_HOLD'
            return SafetyDecision(state=self.state, pause=True)

        if raw_moving and not moving(safe):
            self.hold_started_at = now
            self.clear_count = 0
            self.clear_at = None
            self.state = 'COLLISION_HOLD'
            return SafetyDecision(state=self.state, pause=True)
        self.state = 'COLLISION_SLOWING' if raw_moving and modified else self.phase
        return SafetyDecision(command=safe if raw_moving else Command(), state=self.state)
