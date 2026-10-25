"""
Modern DJI Tello drone controller using djitellopy.

Handles video streaming, keyboard controls, and flight commands.
"""

import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
from djitellopy import Tello

if TYPE_CHECKING:
    # Only needed for type-checking: pynput.keyboard is imported lazily
    # at runtime (see setup_keyboard_controls()) because it performs X11
    # backend detection at import time, which raises ImportError on
    # headless systems (e.g. CI runners) that have no display.
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

# Takeoff can take up to ~20s to be acknowledged (djitellopy's
# TAKEOFF_TIMEOUT), so allow an in-flight command time to finish before
# disconnect() sends anything else to the drone.
FLIGHT_COMMAND_JOIN_TIMEOUT = 25.0

# Key codes cv2.waitKey() returns for named keys. Backspace differs by
# platform (8 on Linux/Windows, 127 on macOS); Enter can be 13 or 10.
# Modifier keys (shift, ctrl, alt) never reach cv2.waitKey() on their own,
# so they can't be used as flight keys.
NAMED_WINDOW_KEY_CODES: Dict[str, Tuple[int, ...]] = {
    "tab": (9,),
    "backspace": (8, 127),
    "esc": (27,),
    "enter": (13, 10),
    "space": (32,),
}

# A movement key held in pynput only counts once the video window has
# also seen that key, proving the window had focus when it was pressed.
# SLACK covers the window seeing the key slightly before pynput does;
# TIMEOUT is how long to wait for the window (whose loop may be slowed
# by inference) before rejecting the press as coming from another app.
WINDOW_CONFIRM_SLACK = 0.25  # seconds
WINDOW_CONFIRM_TIMEOUT = 1.0  # seconds

# cv2.waitKey(1) & 0xFF returns 255 when no key was pressed.
NO_WINDOW_KEY = 255


def window_key_codes(key_name: Optional[str]) -> Tuple[int, ...]:
    """
    Map a config key name (e.g. "tab", "w") to the codes cv2.waitKey() returns.

    Args:
        key_name: Key name from the ``controls`` config section.

    Returns:
        Matching key codes, or an empty tuple if the video window can't
        see this key (e.g. a modifier like "shift").

    """
    if not key_name:
        return ()
    name = str(key_name).lower()
    if name in NAMED_WINDOW_KEY_CODES:
        return NAMED_WINDOW_KEY_CODES[name]
    if len(name) == 1:
        return (ord(name),)
    return ()


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

        # Keyboard input. pynput sees keys system-wide, so it is only used
        # to track held movement keys; each held key must also be seen by
        # the focused video window (see handle_window_key()) before it
        # moves the drone. One-shot flight commands come only from the
        # video window.
        self.listener: Optional["keyboard.Listener"] = None
        self.controls: Dict[str, str] = {}
        self._keys_lock = threading.Lock()
        self._held_keys: Dict[str, float] = {}  # action -> press time
        self._confirmed_keys: Set[str] = set()
        self._rejected_keys: Set[str] = set()
        self._window_key_times: Dict[int, float] = {}  # key code -> last seen

        # Background thread lifecycle management. A single stop event is
        # shared by the video stream and control loops so disconnect() can
        # reliably signal both to exit instead of leaking daemon threads.
        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None

        # Takeoff/land block until the drone answers (up to ~20s), so they
        # run on this thread to keep the video loop - and the emergency
        # key - responsive.
        self._command_thread: Optional[threading.Thread] = None

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

        # Let an in-progress takeoff/land finish first, so its response
        # isn't mixed up with the land command below.
        command_thread = self._command_thread
        if command_thread is not None and command_thread.is_alive():
            print("Waiting for in-progress flight command to finish...")
            command_thread.join(timeout=FLIGHT_COMMAND_JOIN_TIMEOUT)
        self._command_thread = None

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
            Frame as numpy array (BGR) or None if unavailable. This is a
            copy: djitellopy keeps returning the same array until a new
            frame is decoded, so drawing on it directly would stack
            overlays across loop iterations and feed them to the detector.

        """
        try:
            frame: Optional[np.ndarray] = self.drone.get_frame_read().frame
            return None if frame is None else frame.copy()
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

        def stream_loop() -> None:
            while not self._stop_event.is_set():
                frame = self.get_frame()
                if frame is not None and self.frame_callback:
                    self.frame_callback(frame)
                time.sleep(0.01)  # Small delay to prevent CPU hogging

        self._stream_thread = threading.Thread(target=stream_loop, daemon=True)
        self._stream_thread.start()

    def takeoff(self) -> bool:
        """
        Take off the drone.

        Errors (e.g. low battery, no response) are reported rather than
        raised, so a failed takeoff can't take down the caller.

        Returns:
            True if the drone is airborne afterwards.

        """
        if self.is_flying:
            return True

        print("Taking off...")
        try:
            self.drone.takeoff()
        except Exception as e:
            print(f"Takeoff failed: {e}")
            # The command may have timed out even though the drone did
            # take off. If so, track it as flying so it still gets landed.
            if self._reports_airborne():
                print("Drone reports a non-zero height; treating it as airborne.")
                self.is_flying = True
            return self.is_flying

        self.is_flying = True
        print("Airborne!")
        return True

    def land(self) -> bool:
        """
        Land the drone.

        Returns:
            True if the drone is on the ground afterwards. On failure the
            drone is still treated as flying, so a later land() (e.g. from
            disconnect()) tries again.

        """
        if not self.is_flying:
            return True

        print("Landing...")
        try:
            self.drone.land()
        except Exception as e:
            print(f"Landing failed: {e}. The drone may still be airborne!")
            return False

        self.is_flying = False
        print("Landed")
        return True

    def emergency(self) -> None:
        """Emergency stop - cuts motors immediately (the drone will fall)."""
        print("EMERGENCY STOP!")
        try:
            self.drone.emergency()
        except Exception as e:
            print(f"Emergency command failed: {e}")
        self.is_flying = False

    def _reports_airborne(self) -> bool:
        """Best-effort check of the drone's reported height after an error."""
        try:
            return int(self.drone.get_height()) > 0
        except Exception:
            return False

    def _run_flight_command(self, command: Callable[[], bool]) -> None:
        """Run a blocking takeoff/land on the command thread, one at a time."""
        if self._command_thread is not None and self._command_thread.is_alive():
            print("Previous flight command still running; ignoring key.")
            return

        self._command_thread = threading.Thread(
            target=command, name="TelloFlightCommand", daemon=True
        )
        self._command_thread.start()

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
            # cv2-stubs doesn't expose this legacy module-level alias for
            # cv2.VideoWriter.fourcc, even though it exists at runtime.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
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

    def handle_window_key(self, key: int) -> None:
        """
        Handle a key code read from the focused video window.

        Call this every loop iteration with ``cv2.waitKey(1) & 0xFF``.
        Takeoff, land and emergency only come from here, so they only fire
        while the video window has focus. Each key seen here also confirms
        any matching movement key held in pynput.

        Args:
            key: Key code from cv2.waitKey() (255 means no key).

        """
        if key == NO_WINDOW_KEY or key < 0:
            return

        with self._keys_lock:
            self._window_key_times[key] = time.time()

        if key in window_key_codes(self.controls.get("emergency")):
            self.emergency()
        elif key in window_key_codes(self.controls.get("takeoff")):
            self._run_flight_command(self.takeoff)
        elif key in window_key_codes(self.controls.get("land")):
            self._run_flight_command(self.land)

    def _key_to_action(self, key: Any) -> Optional[str]:
        """Map a pynput key to a configured movement action, if any."""
        name = getattr(key, "char", None) or getattr(key, "name", None)
        if not name:
            return None
        name = str(name).lower()
        for action in MOVEMENT_ACTIONS:
            if name == str(self.controls.get(action, "")).lower():
                return action
        return None

    def _on_key_press(self, key: Any) -> None:
        """Record when a movement key goes down (pynput listener thread)."""
        try:
            action = self._key_to_action(key)
            if action is None:
                return
            with self._keys_lock:
                # OS key-repeat sends more presses while held; keep the first.
                self._held_keys.setdefault(action, time.time())
        except Exception as e:  # keep the listener alive whatever happens
            print(f"Keyboard press handler error: {e}")

    def _on_key_release(self, key: Any) -> None:
        """Forget a movement key once it is released (pynput listener thread)."""
        try:
            action = self._key_to_action(key)
            if action is None:
                return
            with self._keys_lock:
                self._held_keys.pop(action, None)
                self._confirmed_keys.discard(action)
                self._rejected_keys.discard(action)
        except Exception as e:  # keep the listener alive whatever happens
            print(f"Keyboard release handler error: {e}")

    def _active_movement_actions(self, now: Optional[float] = None) -> Set[str]:
        """
        Return the held movement actions confirmed by the video window.

        A held key is confirmed once the window has seen one of its key
        codes since (just before) it was pressed. If the window doesn't
        see it within WINDOW_CONFIRM_TIMEOUT, the press came from another
        app and is ignored until the key is released.

        Args:
            now: Current time (defaults to time.time()); for tests.

        Returns:
            Set of confirmed action names.

        """
        if now is None:
            now = time.time()

        with self._keys_lock:
            for action, pressed_at in self._held_keys.items():
                if action in self._confirmed_keys or action in self._rejected_keys:
                    continue
                codes = window_key_codes(self.controls.get(action))
                seen = [
                    self._window_key_times.get(code, float("-inf")) for code in codes
                ]
                if any(t >= pressed_at - WINDOW_CONFIRM_SLACK for t in seen):
                    self._confirmed_keys.add(action)
                elif now - pressed_at > WINDOW_CONFIRM_TIMEOUT:
                    self._rejected_keys.add(action)
            return set(self._confirmed_keys)

    def _compute_rc_vector_from_keys(self, active_keys: Set[str]) -> tuple:
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

        Takeoff/land/emergency are read from the video window via
        handle_window_key(). Held movement keys are tracked with pynput
        (which reports key releases, unlike cv2.waitKey()), but only move
        the drone once the video window confirms it had focus.

        Args:
            controls: Dictionary mapping actions to keys

        """
        self.controls = dict(controls)

        for action in ("takeoff", "land", "emergency", *MOVEMENT_ACTIONS):
            key_name = self.controls.get(action)
            if key_name and not window_key_codes(key_name):
                print(
                    f"Warning: '{action}' key '{key_name}' can't be seen by the "
                    "video window, so it is disabled. Use a letter, digit, "
                    "space, tab, enter, backspace or esc instead."
                )

        # Imported lazily (see the TYPE_CHECKING import above): this
        # requires a real display and is only actually needed here.
        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.listener.start()

        self._stop_event.clear()

        # Control loop for continuous movement. It also acts as a
        # keepalive: while flying it keeps sending (0, 0, 0, 0) so the
        # drone hovers when no key is held.
        def control_loop() -> None:
            while not self._stop_event.is_set():
                active_keys = self._active_movement_actions()
                lr, fb, ud, yaw = self._compute_rc_vector_from_keys(active_keys)
                try:
                    self.send_rc_control(lr, fb, ud, yaw)
                except Exception as e:  # keep the loop (and keepalive) running
                    print(f"RC control error: {e}")

                time.sleep(0.05)

        self._control_thread = threading.Thread(target=control_loop, daemon=True)
        self._control_thread.start()
