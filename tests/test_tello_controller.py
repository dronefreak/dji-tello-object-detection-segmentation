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
