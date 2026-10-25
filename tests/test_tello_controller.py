"""Tests for Tello controller functionality."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from tello_vision.tello_controller import TelloController

# TelloController imports pynput.keyboard lazily inside
# setup_keyboard_controls() because pynput's top-level package
# unconditionally imports both its keyboard and mouse submodules, and
# both perform X11 backend detection at import time - which raises
# ImportError on displayless systems (e.g. CI runners). Patching all
# three into sys.modules before that import runs lets these tests
# exercise the method without a real display or a real global keyboard
# hook.
_FAKE_PYNPUT_KEYBOARD = {
    "pynput": MagicMock(),
    "pynput.keyboard": MagicMock(),
    "pynput.mouse": MagicMock(),
}


class TestTelloController:
    """Tests for TelloController class."""

    def test_controller_creation(self, sample_config):
        """Test creating a controller."""
        with patch("tello_vision.tello_controller.Tello"):
            controller = TelloController(sample_config["drone"])

            assert controller.speed == 50
            assert not controller.is_flying
            assert not controller.is_recording

    def test_get_stats_text(self, sample_config):
        """Test getting formatted stats."""
        with patch("tello_vision.tello_controller.Tello"):
            controller = TelloController(sample_config["drone"])
            controller.battery = 85
            controller.temperature = 45
            controller.height = 120
            controller.flight_time = 65

            stats = controller.get_stats_text()

            assert len(stats) > 0
            assert any("85%" in s for s in stats)
            assert any("45°C" in s for s in stats)

    @patch("tello_vision.tello_controller.Tello")
    def test_connect_success(self, mock_tello_class, sample_config):
        """Test successful connection."""
        mock_drone = Mock()
        mock_drone.get_battery.return_value = 85
        mock_drone.get_temperature.return_value = 45
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        result = controller.connect()

        assert result is True
        mock_drone.connect.assert_called_once()
        mock_drone.streamon.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_connect_failure(self, mock_tello_class, sample_config):
        """Test connection failure."""
        mock_drone = Mock()
        mock_drone.connect.side_effect = Exception("Connection failed")
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        result = controller.connect()

        assert result is False

    @patch("tello_vision.tello_controller.Tello")
    def test_takeoff(self, mock_tello_class, sample_config):
        """Test takeoff command."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.takeoff()

        assert controller.is_flying
        mock_drone.takeoff.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_land(self, mock_tello_class, sample_config):
        """Test land command."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True
        controller.land()

        assert not controller.is_flying
        mock_drone.land.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_emergency(self, mock_tello_class, sample_config):
        """Test emergency stop."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True
        controller.emergency()

        assert not controller.is_flying
        mock_drone.emergency.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_movement_commands(self, mock_tello_class, sample_config):
        """Test various movement commands."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True

        # Test forward
        controller.move_forward(30)
        mock_drone.move_forward.assert_called_with(30)

        # Test backward
        controller.move_back(30)
        mock_drone.move_back.assert_called_with(30)

        # Test left
        controller.move_left(30)
        mock_drone.move_left.assert_called_with(30)

        # Test right
        controller.move_right(30)
        mock_drone.move_right.assert_called_with(30)

        # Test up
        controller.move_up(30)
        mock_drone.move_up.assert_called_with(30)

        # Test down
        controller.move_down(30)
        mock_drone.move_down.assert_called_with(30)

    @patch("tello_vision.tello_controller.Tello")
    def test_rotation_commands(self, mock_tello_class, sample_config):
        """Test rotation commands."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True

        controller.rotate_clockwise(45)
        mock_drone.rotate_clockwise.assert_called_with(45)

        controller.rotate_counter_clockwise(45)
        mock_drone.rotate_counter_clockwise.assert_called_with(45)

    @patch("tello_vision.tello_controller.Tello")
    def test_rc_control(self, mock_tello_class, sample_config):
        """Test RC control."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True

        controller.send_rc_control(10, 20, 30, 40)
        mock_drone.send_rc_control.assert_called_with(10, 20, 30, 40)

    @patch("tello_vision.tello_controller.Tello")
    def test_get_frame(self, mock_tello_class, sample_config):
        """Test getting video frame."""
        mock_drone = Mock()
        mock_frame_read = Mock()
        mock_frame_read.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_drone.get_frame_read.return_value = mock_frame_read
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        frame = controller.get_frame()

        assert frame is not None
        assert frame.shape == (480, 640, 3)

    @patch("tello_vision.tello_controller.Tello")
    @patch("tello_vision.tello_controller.cv2.VideoWriter")
    def test_recording(
        self, mock_writer_class, mock_tello_class, sample_config, tmp_path
    ):
        """Test video recording."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer

        controller = TelloController(sample_config["drone"])

        # Start recording
        output_path = str(tmp_path / "test_video.mp4")
        controller.start_recording(output_path)

        assert controller.is_recording
        assert controller.video_writer is not None

        # Write a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.write_frame(frame)
        mock_writer.write.assert_called_once()

        # Stop recording
        controller.stop_recording()
        assert not controller.is_recording
        mock_writer.release.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_disconnect_cleanup(self, mock_tello_class, sample_config):
        """Test proper cleanup on disconnect."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.is_flying = True

        controller.disconnect()

        # Should land if flying
        mock_drone.land.assert_called()
        mock_drone.streamoff.assert_called()
        mock_drone.end.assert_called()

    @patch("tello_vision.tello_controller.Tello")
    def test_disconnect_stops_video_stream_thread(
        self, mock_tello_class, sample_config
    ):
        """Disconnect() signals the stream thread to stop and joins it."""
        mock_drone = Mock()
        mock_drone.get_frame_read.return_value = Mock(frame=np.zeros((480, 640, 3)))
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.start_video_stream()

        assert controller._stream_thread is not None
        assert controller._stream_thread.is_alive()

        controller.disconnect()

        assert controller._stream_thread is None
        assert controller._stop_event.is_set()

    @patch("tello_vision.tello_controller.Tello")
    def test_disconnect_stops_control_thread(self, mock_tello_class, sample_config):
        """Disconnect() signals the keyboard control thread to stop and joins it."""
        mock_drone = Mock()
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        with patch.dict(sys.modules, _FAKE_PYNPUT_KEYBOARD):
            controller.setup_keyboard_controls(sample_config["controls"])

        assert controller._control_thread is not None
        assert controller._control_thread.is_alive()

        controller.disconnect()

        assert controller._control_thread is None
        assert controller._stop_event.is_set()

    @patch("tello_vision.tello_controller.Tello")
    def test_disconnect_joins_threads_without_hanging(
        self, mock_tello_class, sample_config
    ):
        """Disconnect() returns promptly once both worker threads exit."""
        import time as time_module

        mock_drone = Mock()
        mock_drone.get_frame_read.return_value = Mock(frame=np.zeros((480, 640, 3)))
        mock_tello_class.return_value = mock_drone

        controller = TelloController(sample_config["drone"])
        controller.start_video_stream()
        with patch.dict(sys.modules, _FAKE_PYNPUT_KEYBOARD):
            controller.setup_keyboard_controls(sample_config["controls"])

        start = time_module.time()
        controller.disconnect()
        elapsed = time_module.time() - start

        # Threads poll the stop event and sleep briefly between checks, so
        # shutdown should be fast and well within the join timeout.
        assert elapsed < 2.0
        assert controller._stream_thread is None
        assert controller._control_thread is None


def _make_controller(sample_config, mock_tello_class):
    """Create a controller with a mock drone and the sample key mapping."""
    mock_drone = Mock()
    mock_tello_class.return_value = mock_drone
    controller = TelloController(sample_config["drone"])
    controller.controls = {
        **sample_config["controls"],
        "emergency": "esc",
        "down": "shift",
    }
    return controller, mock_drone


class TestFlightSafety:
    """Regression tests for flight-safety fixes."""

    @patch("tello_vision.tello_controller.Tello")
    def test_get_frame_returns_copy(self, mock_tello_class, sample_config):
        """Drawing on a returned frame must not touch djitellopy's buffer."""
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        buffer = np.zeros((720, 960, 3), dtype=np.uint8)
        mock_drone.get_frame_read.return_value = Mock(frame=buffer)

        frame = controller.get_frame()
        assert frame is not None
        frame[:] = 255

        assert frame is not buffer
        assert buffer.max() == 0

    @patch("tello_vision.tello_controller.Tello")
    def test_takeoff_failure_does_not_raise(self, mock_tello_class, sample_config):
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        mock_drone.takeoff.side_effect = Exception("battery too low")
        mock_drone.get_height.side_effect = Exception("no state")

        assert controller.takeoff() is False
        assert not controller.is_flying

    @patch("tello_vision.tello_controller.Tello")
    def test_takeoff_timeout_while_airborne_is_tracked(
        self, mock_tello_class, sample_config
    ):
        """If takeoff errors but the drone reports height, still land it later."""
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        mock_drone.takeoff.side_effect = Exception("timeout")
        mock_drone.get_height.return_value = 80

        assert controller.takeoff() is True
        assert controller.is_flying

    @patch("tello_vision.tello_controller.Tello")
    def test_land_failure_keeps_flying_state(self, mock_tello_class, sample_config):
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        controller.is_flying = True
        mock_drone.land.side_effect = Exception("no response")

        assert controller.land() is False
        assert controller.is_flying  # so disconnect() retries landing

    @patch("tello_vision.tello_controller.Tello")
    def test_emergency_failure_does_not_raise(self, mock_tello_class, sample_config):
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        controller.is_flying = True
        mock_drone.emergency.side_effect = OSError("socket closed")

        controller.emergency()
        assert not controller.is_flying

    @patch("tello_vision.tello_controller.Tello")
    def test_window_emergency_key_is_immediate(self, mock_tello_class, sample_config):
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        controller.handle_window_key(27)
        mock_drone.emergency.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_window_takeoff_runs_off_main_thread(self, mock_tello_class, sample_config):
        """Takeoff must not block the video loop (and the emergency key)."""
        import threading

        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        release = threading.Event()
        mock_drone.takeoff.side_effect = lambda: release.wait(2.0)

        controller.handle_window_key(9)  # Tab; returns while takeoff blocks
        controller.handle_window_key(27)  # Emergency still goes through
        mock_drone.emergency.assert_called_once()

        release.set()
        assert controller._command_thread is not None
        controller._command_thread.join(timeout=2.0)
        mock_drone.takeoff.assert_called_once()

    @patch("tello_vision.tello_controller.Tello")
    def test_unfocused_keys_do_nothing(self, mock_tello_class, sample_config):
        """Keys from other apps reach pynput only; they must not fly the drone."""
        controller, mock_drone = _make_controller(sample_config, mock_tello_class)
        key_w = Mock(spec=["char"], char="w")

        controller._on_key_press(key_w)
        pressed_at = controller._held_keys["forward"]

        assert controller._active_movement_actions(now=pressed_at) == set()
        # The window never saw it: rejected after the timeout...
        assert controller._active_movement_actions(now=pressed_at + 2.0) == set()
        # ...and stays rejected even if the window later sees a 'w'.
        controller.handle_window_key(ord("w"))
        assert controller._active_movement_actions() == set()

        # Releasing clears it, so the next focused press works again.
        controller._on_key_release(key_w)
        assert controller._held_keys == {}
        mock_drone.takeoff.assert_not_called()

    @patch("tello_vision.tello_controller.Tello")
    def test_focused_movement_key_is_confirmed(self, mock_tello_class, sample_config):
        controller, _ = _make_controller(sample_config, mock_tello_class)
        key_w = Mock(spec=["char"], char="w")

        controller._on_key_press(key_w)
        controller.handle_window_key(ord("w"))

        assert controller._active_movement_actions() == {"forward"}
        controller._on_key_release(key_w)
        assert controller._active_movement_actions() == set()

    @patch("tello_vision.tello_controller.Tello")
    def test_modifier_movement_key_is_never_confirmed(
        self, mock_tello_class, sample_config
    ):
        """Shift can't be seen by the video window, so it can't move the drone."""
        controller, _ = _make_controller(sample_config, mock_tello_class)
        key_shift = Mock(spec=["name"])
        key_shift.name = "shift"

        controller._on_key_press(key_shift)
        controller.handle_window_key(ord("x"))
        assert controller._active_movement_actions() == set()

    @patch("tello_vision.tello_controller.Tello")
    def test_key_handler_errors_do_not_propagate(self, mock_tello_class, sample_config):
        """An exception escaping a pynput callback would stop the listener."""
        controller, _ = _make_controller(sample_config, mock_tello_class)
        controller.controls = Mock(get=Mock(side_effect=RuntimeError("boom")))
        key_w = Mock(spec=["char"], char="w")

        controller._on_key_press(key_w)
        controller._on_key_release(key_w)

    def test_window_key_codes(self):
        from tello_vision.tello_controller import window_key_codes

        assert window_key_codes("tab") == (9,)
        assert window_key_codes("backspace") == (8, 127)
        assert window_key_codes("W") == (ord("w"),)
        assert window_key_codes("shift") == ()
        assert window_key_codes(None) == ()


@pytest.mark.drone
class TestTelloControllerIntegration:
    """Integration tests requiring actual drone."""

    @pytest.mark.skip(reason="Requires actual Tello drone")
    def test_real_connection(self, sample_config):
        """Test connection to real drone."""
        controller = TelloController(sample_config["drone"])
        result = controller.connect()

        if result:
            assert controller.battery > 0
            controller.disconnect()
