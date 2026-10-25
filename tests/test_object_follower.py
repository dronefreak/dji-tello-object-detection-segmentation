"""Tests for the object-follower example's control logic."""

import sys
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
from tello_vision.detectors.base_detector import Detection, DetectionResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import object_follower  # noqa: E402

FRAME_SHAPE = (720, 960, 3)


def _person(cx: int, cy: int = 360, half: int = 40) -> Detection:
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(cx - half, cy - half, cx + half, cy + half),
    )


def _result(*detections: Detection) -> DetectionResult:
    return DetectionResult(list(detections), 0.01, FRAME_SHAPE)


class TestSmoothing:
    def test_update_does_not_modify_detections(self):
        follower = object_follower.ObjectFollower()
        for cx in (100, 200):
            follower.update(_result(_person(cx)), FRAME_SHAPE)

        detection = _person(600)
        follower.update(_result(detection), FRAME_SHAPE)

        assert detection.bbox == (560, 320, 640, 400)

    def test_smoothing_averages_raw_centers(self):
        """Each output is the mean of the last N raw centers, with no feedback."""
        follower = object_follower.ObjectFollower()
        tracked_x = []
        for cx in (100, 200, 300, 400):
            _, tracked = follower.update(_result(_person(cx)), FRAME_SHAPE)
            tracked_x.append(tracked.center[0])

        assert tracked_x == [100, 150, 200, 300]


class TestHandleKey:
    def test_failed_takeoff_keeps_auto_follow_off(self):
        drone = Mock()
        drone.takeoff.return_value = False

        auto_follow, quit_ = object_follower._handle_key(
            object_follower.KEY_TAB, drone, auto_follow=False
        )

        assert (auto_follow, quit_) == (False, False)

    @patch("object_follower.time.sleep")
    def test_successful_takeoff_enables_auto_follow(self, _sleep):
        drone = Mock()
        drone.takeoff.return_value = True

        auto_follow, _ = object_follower._handle_key(
            object_follower.KEY_TAB, drone, auto_follow=False
        )

        assert auto_follow

    def test_macos_backspace_lands(self):
        drone = Mock()
        auto_follow, _ = object_follower._handle_key(
            object_follower.KEY_DELETE, drone, auto_follow=True
        )

        assert not auto_follow
        drone.land.assert_called_once()


@patch("object_follower.time.sleep")
@patch("object_follower.cv2")
@patch("object_follower._initialize")
def test_drone_hovers_when_target_lost(mock_init, mock_cv2, _sleep):
    """The Tello keeps its last RC velocity, so losing the target must send 0s."""
    drone, detector = Mock(), Mock()
    drone.get_frame.return_value = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    drone.takeoff.return_value = True
    visualizer = Mock()
    visualizer.draw_detections.side_effect = lambda frame, _result: frame
    visualizer.draw_stats.side_effect = lambda frame, _stats: frame
    visualizer.draw_crosshair.side_effect = lambda frame: frame
    mock_init.return_value = (drone, detector, visualizer, "person")

    off_center = _result(_person(800))
    detector.detect.side_effect = [off_center, off_center, _result()]
    mock_cv2.waitKey.side_effect = [object_follower.KEY_TAB, 0xFF, ord("p")]

    object_follower.main()

    rc_calls = drone.send_rc_control.call_args_list
    assert len(rc_calls) == 3
    assert rc_calls[0] != call(0, 0, 0, 0)  # following the target
    assert rc_calls[1] == call(0, 0, 0, 0)  # target lost -> hover
    assert rc_calls[2] == call(0, 0, 0, 0)  # shutdown before landing
    drone.land.assert_called_once()


def test_reset_clears_pid_and_history():
    """reset() drops stale PID and smoothing state."""
    follower = object_follower.ObjectFollower()
    follower.update(_result(_person(800)), FRAME_SHAPE)
    follower.pid_yaw._integral = 10.0

    follower.reset()

    assert follower.pid_yaw._integral == 0.0
    assert follower.pid_yaw._prev_time is None
    assert follower.pid_vertical._prev_time is None
    assert len(follower.target_history) == 0


@patch("object_follower.time.sleep")
@patch("object_follower.cv2")
@patch("object_follower._initialize")
def test_follower_resets_when_auto_follow_enabled(mock_init, mock_cv2, _sleep):
    """PIDs run while grounded; enabling follow must start from a clean state."""
    drone, detector, visualizer = Mock(), Mock(), Mock()
    drone.get_frame.return_value = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    drone.takeoff.return_value = True
    visualizer.draw_detections.side_effect = lambda frame, _result: frame
    visualizer.draw_stats.side_effect = lambda frame, _stats: frame
    visualizer.draw_crosshair.side_effect = lambda frame: frame
    mock_init.return_value = (drone, detector, visualizer, "person")
    detector.detect.return_value = _result(_person(800))
    mock_cv2.waitKey.side_effect = [0xFF, object_follower.KEY_TAB, 0xFF, ord("p")]

    with patch.object(object_follower.ObjectFollower, "reset") as mock_reset:
        object_follower.main()

    mock_reset.assert_called_once()
