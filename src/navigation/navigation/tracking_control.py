"""ROS-independent alignment/approach controller.

Call observe() once per *received detection message*, and step() at 10 Hz.
Received-message sequence numbers do not prove that camera images are fresh.
All times passed to this module must use the same monotonic clock.
"""

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Optional


@dataclass(frozen=True)
class TrackingConfig:
    control_period_sec: float = 0.10
    # Vision health is intentionally split: slow != stopped != dead.
    degraded_detection_age_sec: float = 0.50
    max_detection_age_sec: float = 1.00
    degraded_clear_frames: int = 3
    degraded_speed_scale: float = 0.50
    sensor_recovery_timeout_sec: float = 4.0
    sensor_recovery_frames: int = 3
    sensor_recovery_min_interval_sec: float = 0.05
    sensor_recovery_max_interval_sec: float = 1.20
    reacquire_start_frames: int = 3
    lost_abort_frames: int = 12
    lock_target_class: bool = True
    align_reference_x: float = 350.0
    align_tolerance_px: float = 15.0
    align_stable_frames: int = 3
    align_kp: float = 0.0010
    align_min_angular_speed: float = 0.05
    align_max_angular_speed: float = 0.12
    align_timeout_sec: float = 15.0
    # Approach timeout policy: progress watchdog + final absolute safety cap.
    # A slow-but-progressing DEGRADED stream may therefore continue beyond the
    # old fixed 25 s visual timeout, while a stalled robot still fails quickly.
    approach_progress_timeout_sec: float = 12.0
    approach_progress_lower_y_delta_px: float = 5.0
    approach_progress_align_error_delta_px: float = 8.0
    # Hard whole-approach cap including REALIGN, SENSOR_WAIT and final motion.
    approach_timeout_sec: float = 60.0
    approach_stop_lower_y: float = 430.0
    approach_stop_stable_frames: int = 2
    approach_realign_threshold_px: float = 40.0
    approach_steer_kp: float = 0.0004
    approach_max_angular_speed: float = 0.06
    approach_mid_lower_y: float = 330.0
    approach_near_lower_y: float = 390.0
    approach_far_speed: float = 0.08
    approach_mid_speed: float = 0.04
    approach_near_speed: float = 0.02
    # Stop at the visual threshold; do NOT claim that an item was collected.
    stop_only_test_mode: bool = True
    final_approach_calibrated: bool = False
    final_approach_speed: float = 0.03
    # NOMINAL 0.30 m / 0.03 m/s. Not an odometry measurement. Disabled above.
    final_approach_duration_sec: float = 10.0
    tracking_service_timeout_sec: float = 3.0

    def __post_init__(self):
        defaults = {field.name: field.default for field in fields(self)}
        for name, default in defaults.items():
            value = getattr(self, name)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f'{name} must be boolean')
            elif isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be a finite nonnegative number')
        positive = (
            'control_period_sec', 'degraded_detection_age_sec', 'max_detection_age_sec',
            'degraded_speed_scale', 'sensor_recovery_timeout_sec',
            'approach_progress_timeout_sec', 'approach_progress_lower_y_delta_px',
            'approach_progress_align_error_delta_px', 'align_tolerance_px',
            'align_kp', 'align_max_angular_speed', 'align_timeout_sec',
            'approach_timeout_sec', 'approach_stop_lower_y',
            'approach_realign_threshold_px', 'approach_steer_kp',
            'approach_max_angular_speed', 'approach_far_speed',
            'approach_mid_speed', 'approach_near_speed',
            'final_approach_speed', 'tracking_service_timeout_sec',
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be greater than zero')
        if self.control_period_sec > self.degraded_detection_age_sec:
            raise ValueError('control_period_sec must not exceed degraded_detection_age_sec')
        if not self.degraded_detection_age_sec < self.max_detection_age_sec:
            raise ValueError('degraded_detection_age_sec must be smaller than max_detection_age_sec')
        if self.degraded_clear_frames < 1:
            raise ValueError('degraded_clear_frames must be at least 1')
        if not 0 < self.degraded_speed_scale <= 1.0:
            raise ValueError('degraded_speed_scale must be in (0, 1]')
        if self.sensor_recovery_frames < 2:
            raise ValueError('sensor_recovery_frames must be at least 2')
        if not (0 < self.sensor_recovery_min_interval_sec < self.sensor_recovery_max_interval_sec):
            raise ValueError('recovery min interval must be positive and smaller than recovery max interval')
        if self.sensor_recovery_max_interval_sec < self.max_detection_age_sec:
            raise ValueError('sensor_recovery_max_interval_sec must be >= max_detection_age_sec')
        if self.sensor_recovery_timeout_sec <= (
                self.sensor_recovery_frames - 1) * self.sensor_recovery_min_interval_sec:
            raise ValueError('recovery timeout cannot accommodate the recovery streak')
        if self.approach_progress_timeout_sec >= self.approach_timeout_sec:
            raise ValueError('progress timeout must be shorter than total approach timeout')
        if self.reacquire_start_frames >= self.lost_abort_frames:
            raise ValueError('reacquire_start_frames must be smaller than lost_abort_frames')
        if self.align_min_angular_speed > self.align_max_angular_speed:
            raise ValueError('minimum angular speed must not exceed maximum')
        if self.approach_realign_threshold_px <= self.align_tolerance_px:
            raise ValueError('realign threshold must be larger than align tolerance')
        if not (self.approach_mid_lower_y < self.approach_near_lower_y
                < self.approach_stop_lower_y):
            raise ValueError('lower_y thresholds must satisfy mid < near < stop')
        if not (self.approach_near_speed <= self.approach_mid_speed
                <= self.approach_far_speed):
            raise ValueError('approach speeds must satisfy near <= mid <= far')
        if not self.stop_only_test_mode and not self.final_approach_calibrated:
            raise ValueError('Measure final travel and set final_approach_calibrated=true before collection')
        if not self.stop_only_test_mode and self.final_approach_duration_sec <= 0:
            raise ValueError('Set a calibrated final_approach_duration_sec before disabling test mode')
        if self.final_approach_duration_sec >= self.approach_timeout_sec:
            raise ValueError('final duration must be shorter than total approach timeout')


@dataclass(frozen=True)
class Observation:
    sequence: int
    received_at: float
    class_id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def lower_y(self):
        return self.y + self.height / 2.0

    def usable(self, target_class: int, lock_class: bool = True):
        numbers = (self.confidence, self.x, self.y, self.width, self.height)
        return (
            self.class_id >= 0
            and (not lock_class or self.class_id == target_class)
            and all(math.isfinite(value) for value in numbers)
            and 0.0 < self.confidence <= 1.0
            and self.x >= 0.0 and self.y >= 0.0
            and self.width > 0.0 and self.height > 0.0
        )


class VisionHealth(str, Enum):
    NORMAL = 'NORMAL'
    DEGRADED = 'DEGRADED'
    RECOVERY = 'RECOVERY'
    STALE = 'STALE'


class Phase(str, Enum):
    ALIGN = 'ALIGN'
    APPROACH = 'APPROACH'
    REALIGN = 'REALIGN'
    SENSOR_WAIT = 'SENSOR_WAIT'
    FINAL_APPROACH = 'FINAL_APPROACH'
    TEST_COMPLETE = 'TEST_COMPLETE'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'


@dataclass(frozen=True)
class Command:
    linear_x: float = 0.0
    angular_z: float = 0.0


class RecoveryCounter:
    """Count reasonably spaced receptions, not control ticks or callback bursts.

    This is receipt-time health only; no source timestamps are available in
    DetectedObject. A normally spaced but delayed pipeline is not ruled out.
    """
    def __init__(self, required: int = 3, min_interval: float = 0.05,
                 max_interval: float = 0.50):
        if required < 2 or not 0 < min_interval < max_interval:
            raise ValueError('Invalid recovery counter configuration')
        self.required = required
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.reset()

    def reset(self):
        self.count = 0
        self.last_at = None

    def push(self, now: float):
        if not math.isfinite(now):
            self.reset()
            return False
        gap = None if self.last_at is None else now - self.last_at
        if gap is not None and self.min_interval <= gap <= self.max_interval:
            self.count = min(self.required, self.count + 1)
        else:
            self.count = 1
        self.last_at = now
        return self.healthy(now)

    def healthy(self, now: float):
        return (self.last_at is not None and self.count >= self.required
                and 0 <= now - self.last_at <= self.max_interval)


class TrackingController:
    """Single-goal FSM. The ROS adapter serializes observe()/step()/cancel()."""

    def __init__(self, config: TrackingConfig, target_class: int, now: float):
        if target_class < 0:
            raise ValueError('target_class must be a nonnegative class ID')
        self.config = config
        self.target_class = target_class
        self.phase = Phase.ALIGN
        self.reason = ''
        self.started_at = now
        self.alignment_started_at = now
        self.approach_started_at: Optional[float] = None
        self.final_started_at: Optional[float] = None
        self.latest: Optional[Observation] = None
        self.last_sequence = -1
        self.lost_frames = 0
        self.aligned_frames = 0
        self.close_frames = 0
        self.last_valid_lower_y: Optional[float] = None
        # Progress watchdog state. Meaningful visual progress refreshes
        # last_progress_at; merely receiving messages does not.
        self.last_progress_at = now
        self.best_progress_lower_y: Optional[float] = None
        self.best_alignment_error: Optional[float] = None
        self.sensor_wait_started_at: Optional[float] = None
        self.sensor_resume_phase: Optional[Phase] = None
        self.vision_health = VisionHealth.NORMAL
        self.normal_interval_streak = 0
        self.recovery = RecoveryCounter(
            config.sensor_recovery_frames, config.sensor_recovery_min_interval_sec,
            config.sensor_recovery_max_interval_sec,
        )

    @property
    def done(self):
        return self.phase in (
            Phase.SUCCEEDED, Phase.FAILED, Phase.CANCELED, Phase.TEST_COMPLETE,
        )

    @property
    def status(self):
        if self.done:
            return self.phase.value
        if self.phase == Phase.SENSOR_WAIT:
            return f'SENSOR_WAIT/RECOVERY({self.recovery.count}/{self.config.sensor_recovery_frames})'
        if self.phase == Phase.FINAL_APPROACH:
            return self.phase.value
        if self.latest is None:
            return f'{self.phase.value}/WAIT_DETECTION'
        if self.lost_frames:
            near_stop_hold = (
                self.phase == Phase.APPROACH
                and self.last_valid_lower_y is not None
                and self.last_valid_lower_y >= self.config.approach_stop_lower_y
                and self.close_frames > 0
            )
            hold = (
                'NEAR_TARGET_HOLD' if near_stop_hold else
                ('REACQUIRE' if self.lost_frames >= self.config.reacquire_start_frames
                 else 'LOST_HOLD')
            )
            return f'{self.phase.value}/{hold}({self.lost_frames})'
        if self.vision_health == VisionHealth.DEGRADED:
            return f'{self.phase.value}/DEGRADED'
        return self.phase.value

    def fail(self, code: str, detail: str):
        if not self.done:
            if code == 'SENSOR_STALE':
                self.vision_health = VisionHealth.STALE
            self.phase = Phase.FAILED
            self.reason = f'{code}: {detail}'

    def cancel(self):
        # Cancellation wins even between completion and ROS result publication.
        self.phase = Phase.CANCELED
        self.reason = 'STOP'

    def _enter_sensor_wait(self, stale_at: float):
        # Open-loop final motion cannot safely resume based on wall-clock time:
        # a stopped interval must never be counted as completed travel.
        if self.phase == Phase.FINAL_APPROACH:
            self.fail('FINAL_APPROACH_INTERRUPTED',
                      '보완접근 중 수신 중단; 남은 실거리 확인 전 자동 재개 금지')
            return
        self.sensor_resume_phase = self.phase
        self.phase = Phase.SENSOR_WAIT
        self.vision_health = VisionHealth.RECOVERY
        self.normal_interval_streak = 0
        self.sensor_wait_started_at = stale_at
        self.aligned_frames = 0
        self.close_frames = 0
        self.recovery.reset()

    def _record_receive_gap(self, previous_at: Optional[float], now: float):
        """Classify receipt-rate health without turning a slow stream into failure.

        NORMAL is restored only after several consecutive normal intervals to
        avoid flapping around the threshold. Control ticks never increment this
        streak; only new result messages do.
        """
        if previous_at is None:
            return
        gap = now - previous_at
        if gap < 0 or not math.isfinite(gap):
            self.fail('INTERNAL_ERROR', '유효하지 않은 결과 수신 간격')
            return
        if gap > self.config.degraded_detection_age_sec:
            self.vision_health = VisionHealth.DEGRADED
            self.normal_interval_streak = 0
            return
        if self.vision_health == VisionHealth.DEGRADED:
            self.normal_interval_streak += 1
            if self.normal_interval_streak >= self.config.degraded_clear_frames:
                self.vision_health = VisionHealth.NORMAL
                self.normal_interval_streak = 0
        elif self.vision_health == VisionHealth.NORMAL:
            self.normal_interval_streak = 0

    def _refresh_age_health(self, now: float):
        """Enter DEGRADED before the hard stop threshold is reached."""
        if self.done or self.phase == Phase.SENSOR_WAIT:
            return
        last_rx = self.latest.received_at if self.latest else self.started_at
        age = now - last_rx
        if age > self.config.degraded_detection_age_sec:
            self.vision_health = VisionHealth.DEGRADED
            self.normal_interval_streak = 0

    def _mark_progress(self, now: float):
        if math.isfinite(now):
            self.last_progress_at = now

    def _observe_alignment_progress(self, abs_error: float, now: float):
        delta = self.config.approach_progress_align_error_delta_px
        if self.best_alignment_error is None:
            self.best_alignment_error = abs_error
            self._mark_progress(now)
            return
        if abs_error <= self.best_alignment_error - delta:
            self.best_alignment_error = abs_error
            self._mark_progress(now)

    def _observe_approach_progress(self, lower_y: float, now: float):
        delta = self.config.approach_progress_lower_y_delta_px
        if self.best_progress_lower_y is None:
            self.best_progress_lower_y = lower_y
            self._mark_progress(now)
            return
        if lower_y >= self.best_progress_lower_y + delta:
            self.best_progress_lower_y = lower_y
            self._mark_progress(now)

    def _check_deadlines(self, now: float):
        if self.done:
            return
        if not math.isfinite(now) or now < self.started_at:
            self.fail('INTERNAL_ERROR', '유효하지 않은 단조 시각')
            return
        last_rx = self.latest.received_at if self.latest else self.started_at
        self._refresh_age_health(now)
        if (self.phase != Phase.SENSOR_WAIT
                and now - last_rx > self.config.max_detection_age_sec):
            self._enter_sensor_wait(last_rx + self.config.max_detection_age_sec)
            if self.done:
                return
        # The whole-approach cap never resets. The progress watchdog, however,
        # is refreshed only by meaningful geometric progress, not by heartbeats.
        # SENSOR_WAIT and target-loss streaks have their own bounded policies.
        def timeout(code, detail):
            self.fail('SENSOR_STALE' if self.phase == Phase.SENSOR_WAIT else code, detail)

        if (self.phase == Phase.SENSOR_WAIT
                and now - self.sensor_wait_started_at >= self.config.sensor_recovery_timeout_sec):
            self.fail('SENSOR_STALE', '정지 후 수신 정상화 대기 제한 시간 초과')
            return
        if (self.approach_started_at is not None
                and now - self.approach_started_at >= self.config.approach_timeout_sec):
            timeout('APPROACH_TIMEOUT', '재정렬/수신대기/보완접근 포함 절대 접근 제한 시간 초과')
            return
        active_phase = self.sensor_resume_phase if self.phase == Phase.SENSOR_WAIT else self.phase
        if (self.approach_started_at is not None
                and self.phase != Phase.SENSOR_WAIT
                and active_phase != Phase.FINAL_APPROACH
                and self.lost_frames == 0
                and now - self.last_progress_at >= self.config.approach_progress_timeout_sec):
            timeout('APPROACH_TIMEOUT', '유효 검출은 있으나 정렬/접근 진행이 일정 시간 이상 없음')
            return
        if (active_phase in (Phase.ALIGN, Phase.REALIGN)
                and now - self.alignment_started_at >= self.config.align_timeout_sec):
            timeout('ALIGN_TIMEOUT', '정렬 제한 시간 초과')

    def observe(self, observation: Observation):
        """Consume each received result exactly once, including every invalid result."""
        if self.done or observation.sequence <= self.last_sequence:
            return
        if not math.isfinite(observation.received_at):
            self.fail('INTERNAL_ERROR', '유효하지 않은 수신 시각')
            return
        if self.latest is not None and observation.received_at < self.latest.received_at:
            self.fail('INTERNAL_ERROR', '수신 시각 역전')
            return
        previous_at = self.latest.received_at if self.latest is not None else None
        # A late message can enter SENSOR_WAIT, but must not kill a healthy
        # action for a short hiccup or hide a genuine long outage.
        self._check_deadlines(observation.received_at)
        if self.done:
            return
        self.last_sequence = observation.sequence
        self.latest = observation
        if self.phase == Phase.SENSOR_WAIT:
            # Recovery uses its own, deliberately wider interval threshold.
            # A 0.6 s stream can therefore recover even though it is DEGRADED.
            if previous_at is not None:
                gap = observation.received_at - previous_at
                if (math.isfinite(gap) and 0 <= gap <= self.config.degraded_detection_age_sec):
                    self.normal_interval_streak += 1
                else:
                    self.normal_interval_streak = 0
            if not self.recovery.push(observation.received_at):
                return
            # Communication recovered. Independently re-check current target
            # alignment; never immediately carry on with an old forward command.
            previous = self.sensor_resume_phase
            self.phase = (Phase.ALIGN if previous == Phase.ALIGN else Phase.REALIGN)
            if previous == Phase.APPROACH:
                self.alignment_started_at = observation.received_at
            # Preserve the existing ALIGN/REALIGN budget when waiting there.
            self.sensor_wait_started_at = None
            self.sensor_resume_phase = None
            self.vision_health = (
                VisionHealth.NORMAL
                if self.normal_interval_streak >= self.config.degraded_clear_frames
                else VisionHealth.DEGRADED
            )
            self.normal_interval_streak = 0
            self.lost_frames = 0
            self.aligned_frames = 0
            self.close_frames = 0
            self.best_alignment_error = None
            self._mark_progress(observation.received_at)
        else:
            self._record_receive_gap(previous_at, observation.received_at)
            if self.done:
                return
        # Calibrated final motion is intentionally time-limited/open-loop. The
        # object may leave the camera, but message heartbeat is still required.
        if self.phase == Phase.FINAL_APPROACH:
            return
        cfg = self.config
        if not observation.usable(self.target_class, cfg.lock_target_class):
            # If the object vanished immediately after one trustworthy stop-line
            # observation, do not throw away that information.  In stop-only
            # verification mode, reaching the calibrated visual threshold is
            # exactly what the test is meant to prove, so finish safely instead
            # of waiting until LOST_TARGET.  In collection mode, keep the first
            # close confirmation while stopped and require a fresh valid result
            # before allowing final motion.
            near_stop_loss = (
                self.phase == Phase.APPROACH
                and self.last_valid_lower_y is not None
                and self.last_valid_lower_y >= cfg.approach_stop_lower_y
                and self.close_frames > 0
            )
            self.lost_frames += 1
            self.aligned_frames = 0
            if near_stop_loss:
                if cfg.stop_only_test_mode:
                    self.phase = Phase.TEST_COMPLETE
                    self.reason = (
                        'TEST_STOP: 시각 접근 기준 도달 직후 대상 미검출; '
                        '정지 유지, 수거 성공 처리 및 순찰 재개 안 함'
                    )
                    return
                # Preserve the first close confirmation. step() returns zero
                # while lost_frames > 0, so the robot remains stopped. A fresh
                # valid near-threshold result can then complete confirmation.
            else:
                self.close_frames = 0
            if self.lost_frames >= cfg.lost_abort_frames:
                detail = (
                    f'{self.lost_frames}개 결과 연속 미검출/무효 '
                    f'(마지막 lower_y={self.last_valid_lower_y:.1f})'
                    if self.last_valid_lower_y is not None else
                    f'{self.lost_frames}개 결과 연속 미검출/무효'
                )
                self.fail('LOST_TARGET', detail)
            return
        self.lost_frames = 0
        self.last_valid_lower_y = observation.lower_y
        error = cfg.align_reference_x - observation.x
        if self.phase in (Phase.ALIGN, Phase.REALIGN):
            self._observe_alignment_progress(abs(error), observation.received_at)
            self.aligned_frames = self.aligned_frames + 1 if abs(error) <= cfg.align_tolerance_px else 0
            if self.aligned_frames >= cfg.align_stable_frames:
                self.phase = Phase.APPROACH
                if self.approach_started_at is None:
                    self.approach_started_at = observation.received_at
                self.best_progress_lower_y = observation.lower_y
                self.best_alignment_error = None
                self._mark_progress(observation.received_at)
                self.close_frames = 0
            return
        if self.phase == Phase.APPROACH:
            self._observe_approach_progress(observation.lower_y, observation.received_at)
            if (abs(error) >= cfg.approach_realign_threshold_px
                    or (observation.lower_y >= cfg.approach_stop_lower_y
                        and abs(error) > cfg.align_tolerance_px)):
                self.phase = Phase.REALIGN
                self.alignment_started_at = observation.received_at
                self.best_alignment_error = abs(error)
                self.aligned_frames = 0
                self.close_frames = 0
                return
            self.close_frames = (
                self.close_frames + 1
                if observation.lower_y >= cfg.approach_stop_lower_y else 0
            )
            if self.close_frames >= cfg.approach_stop_stable_frames:
                if cfg.stop_only_test_mode:
                    self.phase = Phase.TEST_COMPLETE
                    self.reason = 'TEST_STOP: 시각 접근 기준 도달; 수거 성공 처리 및 순찰 재개 안 함'
                else:
                    self.phase = Phase.FINAL_APPROACH
                    self.final_started_at = observation.received_at

    def _alignment_speed(self, error: float):
        cfg = self.config
        if abs(error) <= cfg.align_tolerance_px:
            return 0.0
        magnitude = min(cfg.align_max_angular_speed,
                        max(cfg.align_min_angular_speed, cfg.align_kp * abs(error)))
        return math.copysign(magnitude, error)

    def _apply_health_scaling(self, command: Command):
        if self.vision_health != VisionHealth.DEGRADED:
            return command
        scale = self.config.degraded_speed_scale
        linear = command.linear_x * scale
        angular = command.angular_z
        if angular:
            if self.phase in (Phase.ALIGN, Phase.REALIGN):
                # Preserve the empirically usable minimum rotation speed while
                # reducing larger commands in a slow-vision state.
                magnitude = max(self.config.align_min_angular_speed, abs(angular) * scale)
                magnitude = min(self.config.align_max_angular_speed, magnitude)
                angular = math.copysign(magnitude, angular)
            else:
                angular *= scale
        return Command(linear_x=linear, angular_z=angular)

    def step(self, now: float):
        """Return desired velocity. Does NOT increment any frame counter."""
        self._check_deadlines(now)
        if self.done or self.phase == Phase.SENSOR_WAIT:
            return Command()
        cfg = self.config
        if self.phase == Phase.FINAL_APPROACH:
            if now - self.final_started_at >= cfg.final_approach_duration_sec:
                self.phase = Phase.SUCCEEDED
                self.reason = 'SUCCESS: 정렬 및 제한 시간 보완접근 완료 (실제 포획 센서 확인 아님)'
                return Command()
            return Command(linear_x=cfg.final_approach_speed)
        obs = self.latest
        if obs is None or self.lost_frames or not obs.usable(self.target_class, cfg.lock_target_class):
            return Command()
        error = cfg.align_reference_x - obs.x
        if self.phase in (Phase.ALIGN, Phase.REALIGN):
            return self._apply_health_scaling(Command(angular_z=self._alignment_speed(error)))
        # Stop on the FIRST near-threshold observation while confirming the
        # second fresh observation. No 1-second blind forward motion remains.
        if obs.lower_y >= cfg.approach_stop_lower_y:
            return Command()
        if abs(error) >= cfg.approach_realign_threshold_px:
            return Command()
        if obs.lower_y < cfg.approach_mid_lower_y:
            speed = cfg.approach_far_speed
        elif obs.lower_y < cfg.approach_near_lower_y:
            speed = cfg.approach_mid_speed
        else:
            speed = cfg.approach_near_speed
        angular = 0.0
        if abs(error) > cfg.align_tolerance_px:
            angular = max(-cfg.approach_max_angular_speed,
                          min(cfg.approach_max_angular_speed, cfg.approach_steer_kp * error))
        return self._apply_health_scaling(Command(linear_x=speed, angular_z=angular))
