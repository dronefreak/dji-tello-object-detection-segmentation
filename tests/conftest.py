"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

# Markers that identify a test as *not* a plain fast unit test. Any test
# collected without one of these already applied is auto-marked "unit" so
# CI's `-m "unit and not slow"` selection actually picks up the test suite
# instead of silently selecting zero tests.
_NON_UNIT_MARKERS = {"integration", "slow", "drone", "gpu"}


def pytest_collection_modifyitems(items):
    """Auto-mark tests as 'unit' unless they already carry another marker.

    Args:
        items: Collected pytest test items.
    """
    for item in items:
        existing_markers = {marker.name for marker in item.iter_markers()}
        if not existing_markers & _NON_UNIT_MARKERS:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def sample_frame():
    """Generate a sample video frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_config():
    """Generate a sample configuration."""
    return {
        "detector": {
            "backend": "yolov8",
            "yolov8": {
                "model": "yolov8n-seg.pt",
                "confidence": 0.5,
                "iou_threshold": 0.45,
                "device": "cpu",
            },
        },
        "drone": {
            "speed": 50,
            "video_bitrate": 4,
            "fps": 30,
            "resolution": [960, 720],
            "connect_retries": 1,
            "connect_retry_delay": 0,
        },
        "visualization": {
            "show_boxes": True,
            "show_masks": True,
            "show_labels": True,
            "show_confidence": True,
            "mask_alpha": 0.4,
            "font_scale": 0.6,
            "font_thickness": 2,
            "box_thickness": 2,
        },
        "processing": {
            "display_window": True,
            "display_fps": True,
            "display_stats": True,
            "frame_skip": 0,
            "output_dir": "./output",
        },
        "controls": {
            "takeoff": "tab",
            "land": "backspace",
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
        },
    }


@pytest.fixture
def mock_detection():
    """Generate a mock detection object."""
    from tello_vision.detectors.base_detector import Detection

    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(100, 100, 300, 400),
        mask=np.ones((480, 640), dtype=np.uint8),
    )


@pytest.fixture
def mock_detection_result(mock_detection):
    """Generate a mock detection result."""
    from tello_vision.detectors.base_detector import DetectionResult

    return DetectionResult(
        detections=[mock_detection], inference_time=0.035, frame_shape=(480, 640, 3)
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
