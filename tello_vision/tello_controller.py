"""Modern DJI Tello drone controller using djitellopy.

Handles video streaming, keyboard controls, and flight commands.
"""

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from djitellopy import Tello
from pynput import keyboard


class TelloController:
    """Controller for DJI Tello drone with video streaming."""

    def __init__(self, config: dict):
        """Initialize Tello controller.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.drone = Tello()

        # State
        self.is_flying = False
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_callback: Optional[Callable[[np.ndarray], None]] = None

        # Stats
        self.battery = 0
        self.temperature = 0
        self.flight_time = 0
        self.height = 0

        # Control settings
        self.speed = config.get("speed", 50)

        # Keyboard listener
        self.listener: Optional[keyboard.Listener] = None
        self.active_keys = set()

    def connect(self) -> bool:
        """Connect to the Tello drone.

        Returns:
            True if connection successful
        """
        try:
            print("Connecting to Tello...")
            self.drone.connect()

            # Get initial state
            self.battery = self.drone.get_battery()
            self.temperature = self.drone.get_temperature()

            print(f"Connected! Battery: {self.battery}%, Temp: {self.temperature}°C")

            # Start video stream
            self.drone.streamon()
            print("Video stream started")

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from drone and cleanup."""
        print("Disconnecting...")

        if self.is_flying:
            self.land()

        if self.is_recording:
            self.stop_recording()

        try:
            self.drone.streamoff()
        except:
            pass

        try:
            self.drone.end()
        except:
            pass

        if self.listener:
            self.listener.stop()

        print("Disconnected")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current video frame from drone.

        Returns:
            Frame as numpy array (BGR) or None if unavailable
        """
        try:
            frame = self.drone.get_frame_read().frame
            return frame
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None

    def start_video_stream(
        self, callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        """Start processing video stream.

        Args:
            callback: Optional callback function to process each frame
        """
        self.frame_callback = callback

        def stream_loop():
            while True:
                frame = self.get_frame()
                if frame is not None and self.frame_callback:
                    self.frame_callback(frame)
                time.sleep(0.01)  # Small delay to prevent CPU hogging

        stream_thread = threading.Thread(target=stream_loop, daemon=True)
        stream_thread.start()

    def takeoff(self) -> None:
        """Take off the drone."""
        if not self.is_flying:
            print("Taking off...")
            self.drone.takeoff()
            self.is_flying = True
            print("Airborne!")

    def land(self) -> None:
        """Land the drone."""
        if self.is_flying:
            print("Landing...")
            self.drone.land()
            self.is_flying = False
            print("Landed")

    def emergency(self) -> None:
        """Emergency stop - cuts motors immediately."""
        print("EMERGENCY STOP!")
        self.drone.emergency()
        self.is_flying = False

    # Movement commands
    def move_forward(self, distance: int = 20) -> None:
        """Move forward (cm)."""
        if self.is_flying:
            self.drone.move_forward(distance)

    def move_back(self, distance: int = 20) -> None:
        """Move backward (cm)."""
        if self.is_flying:
            self.drone.move_back(distance)

    def move_left(self, distance: int = 20) -> None:
        """Move left (cm)."""
        if self.is_flying:
            self.drone.move_left(distance)

    def move_right(self, distance: int = 20) -> None:
        """Move right (cm)."""
        if self.is_flying:
            self.drone.move_right(distance)

    def move_up(self, distance: int = 20) -> None:
        """Move up (cm)."""
        if self.is_flying:
            self.drone.move_up(distance)

    def move_down(self, distance: int = 20) -> None:
        """Move down (cm)."""
        if self.is_flying:
            self.drone.move_down(distance)

    def rotate_clockwise(self, degrees: int = 30) -> None:
        """Rotate clockwise (degrees)."""
        if self.is_flying:
            self.drone.rotate_clockwise(degrees)

    def rotate_counter_clockwise(self, degrees: int = 30) -> None:
        """Rotate counter-clockwise (degrees)."""
        if self.is_flying:
            self.drone.rotate_counter_clockwise(degrees)

    # Continuous control (for smoother movement)
    def send_rc_control(
        self,
        left_right: int = 0,
        forward_backward: int = 0,
        up_down: int = 0,
        yaw: int = 0,
    ) -> None:
        """Send RC control command for smooth movement.

        Args:
            left_right: -100 to 100 (left to right)
            forward_backward: -100 to 100 (backward to forward)
            up_down: -100 to 100 (down to up)
            yaw: -100 to 100 (CCW to CW)
        """
        if self.is_flying:
            self.drone.send_rc_control(left_right, forward_backward, up_down, yaw)

    def update_stats(self) -> None:
        """Update drone telemetry stats."""
        try:
            self.battery = self.drone.get_battery()
            self.temperature = self.drone.get_temperature()
            self.flight_time = self.drone.get_flight_time()
            self.height = self.drone.get_height()
        except:
            pass

    def get_stats_text(self) -> list:
        """Get formatted stats text for display.

        Returns:
            List of stat strings
        """
        return [
            f"Battery: {self.battery}%",
            f"Temp: {self.temperature}°C",
            f"Height: {self.height}cm",
            f"Flight Time: {self.flight_time}s",
            f"Flying: {self.is_flying}",
            f"Recording: {self.is_recording}",
        ]

    def start_recording(
        self, output_path: str, fps: int = 30, resolution: tuple = (960, 720)
    ) -> None:
        """Start recording video.

        Args:
            output_path: Output file path
            fps: Frames per second
            resolution: Video resolution (width, height)
        """
        if not self.is_recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
            self.is_recording = True
            print(f"Recording started: {output_path}")

    def stop_recording(self) -> None:
        """Stop recording video."""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("Recording stopped")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to video file if recording."""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def setup_keyboard_controls(self, controls: dict) -> None:
        """Setup keyboard controls.

        Args:
            controls: Dictionary mapping actions to keys
        """

        def on_press(key):
            try:
                k = key.char if hasattr(key, "char") else key.name

                if k == controls.get("takeoff"):
                    self.takeoff()
                elif k == controls.get("land"):
                    self.land()
                elif k == controls.get("emergency"):
                    self.emergency()
                elif k == controls.get("forward"):
                    self.active_keys.add("forward")
                elif k == controls.get("backward"):
                    self.active_keys.add("backward")
                elif k == controls.get("left"):
                    self.active_keys.add("left")
                elif k == controls.get("right"):
                    self.active_keys.add("right")
                elif k == controls.get("up"):
                    self.active_keys.add("up")
                elif k == controls.get("down"):
                    self.active_keys.add("down")
                elif k == controls.get("yaw_left"):
                    self.active_keys.add("yaw_left")
                elif k == controls.get("yaw_right"):
                    self.active_keys.add("yaw_right")

            except AttributeError:
                pass

        def on_release(key):
            try:
                k = key.char if hasattr(key, "char") else key.name

                # Remove from active keys
                for action in [
                    "forward",
                    "backward",
                    "left",
                    "right",
                    "up",
                    "down",
                    "yaw_left",
                    "yaw_right",
                ]:
                    if k == controls.get(action):
                        self.active_keys.discard(action)

            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        # Start control loop for continuous movement
        def control_loop():
            while True:
                lr = fb = ud = yaw = 0

                if "forward" in self.active_keys:
                    fb = self.speed
                if "backward" in self.active_keys:
                    fb = -self.speed
                if "left" in self.active_keys:
                    lr = -self.speed
                if "right" in self.active_keys:
                    lr = self.speed
                if "up" in self.active_keys:
                    ud = self.speed
                if "down" in self.active_keys:
                    ud = -self.speed
                if "yaw_left" in self.active_keys:
                    yaw = -self.speed
                if "yaw_right" in self.active_keys:
                    yaw = self.speed

                if any([lr, fb, ud, yaw]):
                    self.send_rc_control(lr, fb, ud, yaw)
                else:
                    self.send_rc_control(0, 0, 0, 0)

                time.sleep(0.05)

        control_thread = threading.Thread(target=control_loop, daemon=True)
        control_thread.start()
