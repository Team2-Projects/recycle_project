"""Per-Action velocity gate. The Monitor owns scan, TF and collision geometry.

Request IDs are echoed by the local Humble adapter: a receipt timestamp alone
is never used to associate a safe velocity with a raw intent. No ROS imports.
"""
from dataclasses import dataclass, fields
from itertools import count
import math
import secrets

from navigation.tracking_control import Command

_REQUESTS = count(secrets.randbits(32) << 32)


@dataclass(frozen=True)
class CollisionConfig:
    raw_timeout_sec: float = .50
    safe_timeout_sec: float = .50
    watchdog_period_sec: float = .05
    ready_timeout_sec: float = 3.
    hold_timeout_sec: float = 3.
    hold_clear_ratio: float = .80
    hold_clear_frames: int = 3
    clear_min_interval_sec: float = .04
    linear_zero_threshold: float = .002
    angular_zero_threshold: float = .005
    modification_tolerance: float = 1e-5

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'collision_{field.name} must be finite and positive')
        if type(self.hold_clear_frames) is not int or self.hold_clear_frames < 2:
            raise ValueError('hold_clear_frames must be an integer >= 2')
        if not 0 < self.hold_clear_ratio <= 1:
            raise ValueError('hold_clear_ratio must be in (0, 1]')
        if self.watchdog_period_sec >= min(self.raw_timeout_sec, self.safe_timeout_sec):
            raise ValueError('watchdog must run before freshness deadlines')
        if self.hold_timeout_sec <= (self.hold_clear_frames - 1) * self.clear_min_interval_sec:
            raise ValueError('hold timeout must accommodate the clear streak')


@dataclass(frozen=True)
class SafetyDecision:
    command: Command = Command()
    state: str = 'IDLE'
    pause: bool = False
    realign: bool = False
    failure: str = ''


def moving(command):
    return abs(command.linear_x) > 1e-9 or abs(command.angular_z) > 1e-9


def compatible(raw, safe, tolerance):
    """Only a finite same-direction scalar reduction; never enlarge a component."""
    ratios = []
    for requested, returned in ((raw.linear_x, safe.linear_x), (raw.angular_z, safe.angular_z)):
        if not math.isfinite(requested) or not math.isfinite(returned):
            return False
        if abs(returned) > abs(requested) or requested * returned < 0:
            return False
        if requested:
            ratios.append(returned / requested)
    return len(ratios) < 2 or abs(ratios[0] - ratios[1]) <= tolerance


class CollisionSafety:
    """Freshness, command ownership and one obstacle HOLD; no health recovery FSM."""
    def __init__(self, config):
        self.config = config
        self.raw = self.safe = Command()
        self.raw_at = self.safe_at = self.pending_at = None
        self.phase = self.state = 'IDLE'
        self.input_valid = False
        self.failure = self.last_fault_reason = ''
        self.hold_started_at = self.clear_at = None
        self.clear_count = self.safe_seq = self.processed_seq = 0
        self.final_armed = False
        self._requests = {}
        self._last_reply_id = -1

    @property
    def suspended(self):
        return self.hold_started_at is not None

    def request(self, command, phase, now, force=False):
        if not math.isfinite(now) or not compatible(command, command, self.config.modification_tolerance):
            self.fail('SAFETY_INVALID_COMMAND')
        changed = command != self.raw or phase != self.phase
        gap = self.raw_at is None or not 0 <= now - self.raw_at <= self.config.raw_timeout_sec
        self.raw, self.raw_at, self.phase = command, now, phase
        if changed or gap or force:
            self._requests.clear()
            self.safe_at = None
            self.clear_count, self.clear_at = 0, None
        if self.safe_at is None and self.pending_at is None:
            self.pending_at = now
        self._requests = {key: at for key, at in self._requests.items()
                          if 0 <= now - at <= self.config.safe_timeout_sec}
        request_id = next(_REQUESTS)
        self._requests[request_id] = now
        return request_id

    def accept_safe(self, command, now, request_id, input_valid=True):
        sent = self._requests.pop(request_id, None)
        if (self.failure or sent is None or request_id <= self._last_reply_id
                or not 0 <= now - sent <= self.config.safe_timeout_sec):
            return False
        if not compatible(self.raw, command, self.config.modification_tolerance):
            self.fail('SAFETY_INVALID_OUTPUT')
            return False
        self.safe, self.safe_at, self.input_valid = command, now, input_valid
        self._last_reply_id = request_id
        self.safe_seq += 1
        self.pending_at = None
        return True

    def fail(self, reason):
        self.failure = self.failure or reason
        self.state = 'FAILED'

    def mark_final_started(self):
        self.final_armed = True

    def acknowledge_realign(self, now):
        if self.clear_count < self.config.hold_clear_frames or self.failure:
            return False
        self.hold_started_at = self.clear_at = self.safe_at = None
        self.clear_count = 0
        self._requests.clear()
        self.pending_at = now
        return True

    def evaluate(self, now):
        if self.failure:
            return SafetyDecision(state='FAILED', pause=True, failure=self.failure)
        cfg = self.config
        reason = ''
        if not math.isfinite(now):
            reason = 'CLOCK_ERROR'
        elif self.raw_at is None or not 0 <= now - self.raw_at <= cfg.raw_timeout_sec:
            reason = 'RAW_STALE'
        elif self.safe_at is None:
            if self.final_armed or self.pending_at is None or now - self.pending_at > cfg.safe_timeout_sec:
                reason = 'SAFE_STALE'
            elif not self.failure:
                return SafetyDecision(state='WAIT_SAFE', pause=self.suspended)
        elif not 0 <= now - self.safe_at <= cfg.safe_timeout_sec:
            reason = 'SAFE_STALE'
        elif not self.input_valid:
            reason = 'MONITOR_INPUT_UNAVAILABLE'
        if reason:
            self.last_fault_reason = reason
            self.fail('FINAL_APPROACH_INTERRUPTED' if self.final_armed else 'SAFETY_UNAVAILABLE')
        if self.failure:
            return SafetyDecision(state='FAILED', pause=True, failure=self.failure)
        safe = self.safe
        if abs(safe.linear_x) < cfg.linear_zero_threshold and abs(safe.angular_z) < cfg.angular_zero_threshold:
            safe = Command()
        modified = (abs(self.raw.linear_x - safe.linear_x) > cfg.modification_tolerance
                    or abs(self.raw.angular_z - safe.angular_z) > cfg.modification_tolerance)
        if self.phase == 'FINAL_APPROACH' and moving(self.raw) and modified:
            self.fail('FINAL_APPROACH_INTERRUPTED')
            return SafetyDecision(state='FAILED', pause=True, failure=self.failure)
        if moving(self.raw) and not moving(safe) and self.hold_started_at is None:
            self.hold_started_at = now
        if self.hold_started_at is not None:
            if now - self.hold_started_at >= cfg.hold_timeout_sec:
                self.fail('COLLISION_BLOCKED')
                return SafetyDecision(state='FAILED', pause=True, failure=self.failure)
            if self.safe_seq != self.processed_seq:
                self.processed_seq = self.safe_seq
                clear = moving(self.raw) and all(
                    not raw or abs(val) >= abs(raw) * cfg.hold_clear_ratio
                    for raw, val in ((self.raw.linear_x, safe.linear_x), (self.raw.angular_z, safe.angular_z)))
                if not clear:
                    self.clear_count, self.clear_at = 0, None
                elif self.clear_at is None or now - self.clear_at >= cfg.clear_min_interval_sec:
                    self.clear_count += 1
                    self.clear_at = now
            realign = self.clear_count >= cfg.hold_clear_frames
            self.state = 'REALIGN_REQUIRED' if realign else 'COLLISION_HOLD'
            return SafetyDecision(state=self.state, pause=True, realign=realign)
        self.state = 'COLLISION_SLOWING' if modified else 'CLEAR'
        return SafetyDecision(command=safe if moving(self.raw) else Command(), state=self.state)
