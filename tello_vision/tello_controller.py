"""
Modern DJI Tello drone controller using djitellopy.

Handles video streaming, keyboard controls, and flight commands.
"""

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from djitellopy import Tello
from pynput import keyboard

# Actions that map to continuous (held-key) movement, as opposed to the
# one-shot takeoff/land/emergency actions.
MOVEMENT_ACTIONS = (
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
    "yaw_left",
    "yaw_right",
)

# Default connection retry behavior. Tello WiFi connections are commonly
# flaky, so a single failed attempt shouldn't abort the whole application.
DEFAULT_CONNECT_RETRIES = 3
DEFAULT_CONNECT_RETRY_DELAY = 1.0  # seconds, doubles after each attempt
MAX_CONNECT_RETRY_DELAY = 10.0  # cap for exponential backoff

# How long to wait for background threads to exit on disconnect().
THREAD_JOIN_TIMEOUT = 2.0


class TelloController:
    """Controller for DJI Tello drone with video streaming."""

    def __init__(self, config: dict):
        """
        Initialize Tello controller.

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

        # Connection retry settings
        self.connect_retries = config.get("connect_retries", DEFAULT_CONNECT_RETRIES)
        self.connect_retry_delay = config.get(
            "connect_retry_delay", DEFAULT_CONNECT_RETRY_DELAY
        )

        # Keyboard listener
        self.listener: Optional[keyboard.Listener] = None
        self.active_keys = set()
        self._active_keys_lock = threading.Lock()

        # Background thread lifecycle management. A single stop event is
        # shared by the video stream and control loops so disconnect() can
        # reliably signal both to exit instead of leaking daemon threads.
        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """
        Connect to the Tello drone, retrying transient failures.

        Tello WiFi connections are commonly flaky, so this retries with
        exponential backoff (configurable via ``connect_retries`` and
        ``connect_retry_delay`` in the drone config) instead of aborting
        on the first failure.

        Returns:
            True if connection successful

        """
        attempts = max(1, self.connect_retries)
        delay = self.connect_retry_delay

        for attempt in range(1, attempts + 1):
            try:
                print(f"Connecting to Tello (attempt {attempt}/{attempts})...")
                self.drone.connect()

                # Get initial state
                self.battery = self.drone.get_battery()
                self.temperature = self.drone.get_temperature()

                print(
                    f"Connected! Battery: {self.battery}%, Temp: {self.temperature}°C"
                )

                # Start video stream
                self.drone.streamon()
                print("Video stream started")

                return True

            except Exception as e:
                print(f"Connection attempt {attempt}/{attempts} failed: {e}")
                if attempt < attempts:
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_CONNECT_RETRY_DELAY)

        print("Failed to connect after all retry attempts")
        return False

    def disconnect(self) -> None:
        """Disconnect from drone and cleanup."""
        print("Disconnecting...")

        # Signal background threads (stream/control loops) to stop and
        # wait for them to actually exit before tearing down the
        # connection, to avoid races with self.drone.end()/streamoff().
        self._stop_event.set()
        for thread in (self._stream_thread, self._control_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
        self._stream_thread = None
        self._control_thread = None

        if self.is_flying:
            self.land()

        if self.is_recording:
            self.stop_recording()

        try:
            self.drone.streamoff()
        except Exception as e:
            print(f"Error stopping stream: {e}")

        try:
            self.drone.end()
        except Exception as e:
            print(f"Error ending connection: {e}")

        if self.listener:
            self.listener.stop()

        print("Disconnected")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current video frame from drone.

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
        """
        Start processing video stream.

        Args:
            callback: Optional callback function to process each frame

        """
        self.frame_callback = callback
        self._stop_event.clear()

        def stream_loop():
            while not self._stop_event.is_set():
                frame = self.get_frame()
                if frame is not None and self.frame_callback:
                    self.frame_callback(frame)
                time.sleep(0.01)  # Small delay to prevent CPU hogging

        self._stream_thread = threading.Thread(target=stream_loop, daemon=True)
        self._stream_thread.start()

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
        """
        Send RC control command for smooth movement.

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
        except Exception as e:
            print(f"Error updating stats: {e}")

    def get_stats_text(self) -> list:
        """
        Get formatted stats text for display.

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
        """
        Start recording video.

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

    def _compute_rc_vector_from_keys(self, active_keys: set) -> tuple:
        """
        Translate currently-held movement keys into an RC control vector.

        Args:
            active_keys: Set of currently active movement action names.

        Returns:
            (lr, fb, ud, yaw) RC control values.

        """
        lr = fb = ud = yaw = 0

        if "forward" in active_keys:
            fb = self.speed
        if "backward" in active_keys:
            fb = -self.speed
        if "left" in active_keys:
            lr = -self.speed
        if "right" in active_keys:
            lr = self.speed
        if "up" in active_keys:
            ud = self.speed
        if "down" in active_keys:
            ud = -self.speed
        if "yaw_left" in active_keys:
            yaw = -self.speed
        if "yaw_right" in active_keys:
            yaw = self.speed

        return lr, fb, ud, yaw

    def setup_keyboard_controls(self, controls: dict) -> None:
        """
        Set up keyboard controls.

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
                else:
                    for action in MOVEMENT_ACTIONS:
                        if k == controls.get(action):
                            with self._active_keys_lock:
                                self.active_keys.add(action)
                            break

            except AttributeError:
                pass

        def on_release(key):
            try:
                k = key.char if hasattr(key, "char") else key.name

                # Remove from active keys
                for action in MOVEMENT_ACTIONS:
                    if k == controls.get(action):
                        with self._active_keys_lock:
                            self.active_keys.discard(action)

            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        self._stop_event.clear()

        # Start control loop for continuous movement
        def control_loop():
            while not self._stop_event.is_set():
                with self._active_keys_lock:
                    active_keys = set(self.active_keys)

                lr, fb, ud, yaw = self._compute_rc_vector_from_keys(active_keys)
                self.send_rc_control(lr, fb, ud, yaw)

                time.sleep(0.05)

        self._control_thread = threading.Thread(target=control_loop, daemon=True)
        self._control_thread.start()
