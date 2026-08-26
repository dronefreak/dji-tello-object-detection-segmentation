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
from typing import Optional

import cv2
import numpy as np
import yaml

from tello_vision.detectors.base_detector import BaseDetector
from tello_vision.tello_controller import TelloController
from tello_vision.visualizer import Visualizer


class PIDController:
    """A standard PID controller with output clamping and anti-windup.

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
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = (
            integral_limit if integral_limit is not None else output_limit
        )

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self) -> None:
        """Reset accumulated state (integral/derivative history)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        """Compute the next control output for the given error.

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
    """PID-based object follower.

    Keeps the target object centered in frame.
    """

    def __init__(self, target_class: str = "person"):
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
        self.target_history = deque(maxlen=5)  # Smooth tracking
        self.lost_frames = 0
        self.max_lost_frames = 30

    def find_target(self, result, frame_shape):
        """Find the best target in detections."""
        candidates = [d for d in result.detections if d.class_name == self.target_class]

        if not candidates:
            return None

        # Choose largest detection (closest object)
        return max(candidates, key=lambda d: d.area)

    def calculate_control(self, target, frame_shape):
        """Calculate control commands based on target position.

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

    def update(self, result, frame_shape):
        """Update tracking and return control commands.

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
            if len(self.target_history) >= 3:
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


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("Initializing...")
    drone = TelloController(config["drone"])

    detector_config = config["detector"]
    backend = detector_config["backend"]
    detector = BaseDetector.create_detector(backend, detector_config[backend])
    detector.load_model()
    detector.warmup()

    visualizer = Visualizer(config["visualization"])

    # Initialize follower
    target_class = input("Enter target class to follow (default: person): ").strip()
    if not target_class:
        target_class = "person"

    follower = ObjectFollower(target_class=target_class)

    # Connect to drone
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
            # Get frame
            frame = drone.get_frame()
            if frame is None:
                continue

            # Run detection
            result = detector.detect(frame)

            # Update follower
            control, target = follower.update(result, frame.shape)

            # Execute control if auto-follow enabled
            if auto_follow and control:
                lr, fb, ud, yaw = control
                drone.send_rc_control(lr, fb, ud, yaw)

                # Draw target indicator
                if target:
                    cx, cy = target.center
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 3)
                    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
                    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

            # Visualize
            frame = visualizer.draw_detections(frame, result)

            # Draw status
            status = [
                f"Auto-Follow: {'ON' if auto_follow else 'OFF'}",
                f"Target: {target_class}",
                f"Tracking: {'YES' if control else 'NO'}",
                f"Lost Frames: {follower.lost_frames}",
            ]

            if control:
                lr, fb, ud, yaw = control
                status.extend(
                    [f"LR: {lr:+3d}  FB: {fb:+3d}", f"UD: {ud:+3d}  YAW: {yaw:+3d}"]
                )

            frame = visualizer.draw_stats(frame, status)

            # Draw crosshair
            frame = visualizer.draw_crosshair(frame)

            # Display
            cv2.imshow("Object Following", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == 9:  # TAB
                if not auto_follow:
                    drone.takeoff()
                    time.sleep(3)  # Wait for stable hover
                    auto_follow = True
                    print("Auto-follow ENABLED")

            elif key == 8:  # BACKSPACE
                if auto_follow:
                    auto_follow = False
                    drone.send_rc_control(0, 0, 0, 0)  # Stop movement
                    drone.land()
                    print("Auto-follow DISABLED")

            elif key == 27:  # ESC
                drone.emergency()
                break

            elif key == ord("p"):
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
