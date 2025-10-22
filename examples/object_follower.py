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

import cv2
import numpy as np
import yaml

from tello_vision.detectors.base_detector import BaseDetector
from tello_vision.tello_controller import TelloController
from tello_vision.visualizer import Visualizer


class ObjectFollower:
    """Simple PID-based object follower.

    Keeps the target object centered in frame.
    """

    def __init__(self, target_class: str = "person"):
        self.target_class = target_class

        # PID parameters (tune these)
        self.kp_yaw = 0.5
        self.kp_forward = 0.3
        self.kp_vertical = 0.4

        # Target area thresholds
        self.min_area = 5000  # Too far, move forward
        self.max_area = 50000  # Too close, move back

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
        yaw = int(self.kp_yaw * error_x / w * 100)
        yaw = np.clip(yaw, -50, 50)

        # Forward/backward control (maintain distance)
        if target_area < self.min_area:
            fb = 20  # Move forward
        elif target_area > self.max_area:
            fb = -20  # Move back
        else:
            fb = 0

        # Vertical control (keep centered vertically)
        ud = int(self.kp_vertical * error_y / h * 100)
        ud = np.clip(ud, -30, 30)

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

            # Use smoothed position
            if len(self.target_history) >= 3:
                _ = np.mean([t.center for t in self.target_history], axis=0).astype(int)
                target.bbox = (
                    target.bbox[0],
                    target.bbox[1],
                    target.bbox[2],
                    target.bbox[3],
                )

            return self.calculate_control(target, frame_shape), target
        else:
            self.lost_frames += 1

            if self.lost_frames > self.max_lost_frames:
                self.target_history.clear()

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
