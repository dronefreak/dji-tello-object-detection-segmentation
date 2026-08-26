"""
Example: Object tracking and following with Tello.

Demonstrates autonomous behavior - drone follows a detected person.

This is a starting point for self-driving car concepts applied to drones:
- Object detection
- Target tracking
- Reactive control based on object position
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from tello_vision.detectors.base_detector import (
    BaseDetector,
    Detection,
    DetectionResult,
)
from tello_vision.tello_controller import TelloController
from tello_vision.visualizer import Visualizer

# (lr, fb, ud, yaw) RC control values, as returned by ObjectFollower.update().
ControlVector = Tuple[int, int, int, int]

# Number of recent target positions to average for jitter smoothing.
SMOOTHING_WINDOW = 3

# cv2.waitKey() key codes used for interactive controls.
KEY_TAB = 9
KEY_BACKSPACE = 8
KEY_ESC = 27


class PIDController:
    """
    A standard PID controller with output clamping and anti-windup.

    Anti-windup is implemented by clamping the accumulated integral term
    to ``integral_limit`` so a persistently large error (e.g. the target
    briefly leaving the frame) can't cause the integral term to grow
    unbounded and produce a large control overshoot once the error is
    corrected.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
        integral_limit: Optional[float] = None,
    ):
        """
        Initialize the PID controller with gains and output limits.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            output_limit: Maximum absolute output value.
            integral_limit: Maximum absolute accumulated integral, used to
                clamp anti-windup. Defaults to output_limit if not given.

        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = (
            integral_limit if integral_limit is not None else output_limit
        )

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        """Reset accumulated state (integral/derivative history)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        """
        Compute the next control output for the given error.

        Args:
            error: Current error (setpoint - measured value).

        Returns:
            Control output, clamped to +/- output_limit.

        """
        now = time.time()
        dt = now - self._prev_time if self._prev_time is not None else 0.0
        self._prev_time = now

        self._integral += error * dt
        self._integral = float(
            np.clip(self._integral, -self.integral_limit, self.integral_limit)
        )

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


class ObjectFollower:
    """
    PID-based object follower.

    Keeps the target object centered in frame.
    """

    def __init__(self, target_class: str = "person"):
        """
        Initialize the follower for a given target detection class.

        Args:
            target_class: Class name of the object to follow (e.g. "person").

        """
        self.target_class = target_class

        # PID controllers for yaw (horizontal centering) and vertical
        # (up/down centering). Gains are tuned conservatively; adjust as
        # needed for your drone/environment.
        self.pid_yaw = PIDController(kp=0.5, ki=0.05, kd=0.1, output_limit=50)
        self.pid_vertical = PIDController(kp=0.4, ki=0.02, kd=0.08, output_limit=30)

        # Forward/backward is distance-based (bang-bang on bbox area)
        # rather than PID, since we don't have a continuous depth signal.
        # Target area thresholds
        self.min_area = 5000  # Too far, move forward
        self.max_area = 50000  # Too close, move back
        self.forward_speed = 20

        # Tracking state
        self.target_history: deque = deque(maxlen=5)  # Smooth tracking
        self.lost_frames = 0
        self.max_lost_frames = 30

    def find_target(
        self, result: DetectionResult, frame_shape: Tuple[int, ...]
    ) -> Optional[Detection]:
        """Find the best target in detections."""
        candidates = [d for d in result.detections if d.class_name == self.target_class]

        if not candidates:
            return None

        # Choose largest detection (closest object)
        return max(candidates, key=lambda d: d.area)

    def calculate_control(
        self, target: Detection, frame_shape: Tuple[int, ...]
    ) -> ControlVector:
        """
        Calculate control commands based on target position.

        Returns:
            (lr, fb, ud, yaw) control values

        """
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2

        target_x, target_y = target.center
        target_area = target.area

        # Calculate errors
        error_x = target_x - center_x
        error_y = center_y - target_y  # Inverted (down is positive)

        # Yaw control (keep centered horizontally)
        yaw = int(self.pid_yaw.update(error_x / w * 100))

        # Forward/backward control (maintain distance)
        if target_area < self.min_area:
            fb = self.forward_speed  # Move forward
        elif target_area > self.max_area:
            fb = -self.forward_speed  # Move back
        else:
            fb = 0

        # Vertical control (keep centered vertically)
        ud = int(self.pid_vertical.update(error_y / h * 100))

        # No left/right movement (use yaw instead)
        lr = 0

        return lr, fb, ud, yaw

    def update(
        self, result: DetectionResult, frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[ControlVector], Optional[Detection]]:
        """
        Update tracking and return control commands.

        Returns:
            (lr, fb, ud, yaw) or None if target lost

        """
        target = self.find_target(result, frame_shape)

        if target:
            self.target_history.append(target)
            self.lost_frames = 0

            # Use smoothed position to reduce jitter in the control loop.
            # Average the recent detection centers, then re-center the
            # current bbox on that smoothed point (keeping its size).
            if len(self.target_history) >= SMOOTHING_WINDOW:
                smoothed_cx, smoothed_cy = np.mean(
                    [t.center for t in self.target_history], axis=0
                ).astype(int)
                x1, y1, x2, y2 = target.bbox
                half_width, half_height = (x2 - x1) // 2, (y2 - y1) // 2
                target.bbox = (
                    int(smoothed_cx - half_width),
                    int(smoothed_cy - half_height),
                    int(smoothed_cx + half_width),
                    int(smoothed_cy + half_height),
                )

            return self.calculate_control(target, frame_shape), target
        else:
            self.lost_frames += 1

            if self.lost_frames > self.max_lost_frames:
                self.target_history.clear()
                # Reset PID state so stale integral/derivative history
                # doesn't cause a control spike when the target reappears.
                self.pid_yaw.reset()
                self.pid_vertical.reset()

            return None, None


def _initialize() -> Tuple[TelloController, BaseDetector, Visualizer, str]:
    """
    Load config and construct drone/detector/visualizer components.

    Returns:
        Tuple of (drone, detector, visualizer, target_class).

    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing...")
    drone = TelloController(config["drone"])

    detector_config = config["detector"]
    backend = detector_config["backend"]
    detector = BaseDetector.create_detector(backend, detector_config[backend])
    detector.load_model()
    detector.warmup()

    visualizer = Visualizer(config["visualization"])

    target_class = input("Enter target class to follow (default: person): ").strip()
    if not target_class:
        target_class = "person"

    return drone, detector, visualizer, target_class


def _handle_key(key: int, drone: TelloController, auto_follow: bool) -> tuple:
    """
    Handle a single keypress from the main loop.

    Args:
        key: Key code from cv2.waitKey().
        drone: Active drone controller.
        auto_follow: Current auto-follow state.

    Returns:
        (new_auto_follow, should_quit) tuple.

    """
    if key == KEY_TAB:
        if not auto_follow:
            drone.takeoff()
            time.sleep(3)  # Wait for stable hover
            auto_follow = True
            print("Auto-follow ENABLED")
    elif key == KEY_BACKSPACE:
        if auto_follow:
            auto_follow = False
            drone.send_rc_control(0, 0, 0, 0)  # Stop movement
            drone.land()
            print("Auto-follow DISABLED")
    elif key == KEY_ESC:
        drone.emergency()
        return auto_follow, True
    elif key == ord("p"):
        return auto_follow, True

    return auto_follow, False


@dataclass
class FrameState:
    """Per-frame control/tracking context passed to `_draw_frame`."""

    control: Optional[ControlVector]
    target: Optional[Detection]
    auto_follow: bool
    follower: "ObjectFollower"
    target_class: str


def _draw_frame(
    visualizer: Visualizer,
    frame: np.ndarray,
    result: DetectionResult,
    frame_state: "FrameState",
) -> np.ndarray:
    """
    Draw detections, tracking indicator, and status overlays on a frame.

    Args:
        visualizer: Visualizer instance.
        frame: Current video frame.
        result: Detection result for this frame.
        frame_state: FrameState with the current control/tracking context.

    Returns:
        The annotated frame.

    """
    control = frame_state.control
    target = frame_state.target
    auto_follow = frame_state.auto_follow
    follower = frame_state.follower
    target_class = frame_state.target_class

    if auto_follow and control and target:
        cx, cy = target.center
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 3)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

    frame = visualizer.draw_detections(frame, result)

    status = [
        f"Auto-Follow: {'ON' if auto_follow else 'OFF'}",
        f"Target: {target_class}",
        f"Tracking: {'YES' if control else 'NO'}",
        f"Lost Frames: {follower.lost_frames}",
    ]
    if control:
        lr, fb, ud, yaw = control
        status.extend([f"LR: {lr:+3d}  FB: {fb:+3d}", f"UD: {ud:+3d}  YAW: {yaw:+3d}"])

    frame = visualizer.draw_stats(frame, status)
    return visualizer.draw_crosshair(frame)


def main() -> None:
    """Run the interactive object-following demo."""
    drone, detector, visualizer, target_class = _initialize()
    follower = ObjectFollower(target_class=target_class)

    if not drone.connect():
        print("Failed to connect to drone")
        return

    print("\n" + "=" * 60)
    print("Object Following Mode")
    print("=" * 60)
    print(f"Target: {target_class}")
    print("\nControls:")
    print("  TAB: Enable auto-follow (drone will takeoff)")
    print("  BACKSPACE: Disable auto-follow (drone will land)")
    print("  ESC: Emergency stop")
    print("  P: Quit")
    print("=" * 60 + "\n")

    auto_follow = False

    try:
        while True:
            frame = drone.get_frame()
            if frame is None:
                continue

            result = detector.detect(frame)
            control, target = follower.update(result, frame.shape)

            if auto_follow and control:
                lr, fb, ud, yaw = control
                drone.send_rc_control(lr, fb, ud, yaw)

            frame = _draw_frame(
                visualizer,
                frame,
                result,
                FrameState(control, target, auto_follow, follower, target_class),
            )
            cv2.imshow("Object Following", frame)

            key = cv2.waitKey(1) & 0xFF
            auto_follow, should_quit = _handle_key(key, drone, auto_follow)
            if should_quit:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        if auto_follow:
            drone.send_rc_control(0, 0, 0, 0)
            drone.land()

        cv2.destroyAllWindows()
        drone.disconnect()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
