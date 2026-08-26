"""Tests for the asynchronous inference worker."""

import time
from unittest.mock import Mock

import numpy as np
from tello_vision.detectors.base_detector import DetectionResult
from tello_vision.inference_worker import AsyncInferenceWorker


def _make_result(count: int = 0) -> DetectionResult:
    return DetectionResult(
        detections=[], inference_time=0.001, frame_shape=(480, 640, 3)
    )


class TestAsyncInferenceWorker:
    """Tests for AsyncInferenceWorker."""

    def test_no_result_before_any_frame_submitted(self):
        """get_latest_result() returns None until inference has run once."""
        detector = Mock()
        detector.detect.return_value = _make_result()

        worker = AsyncInferenceWorker(detector)
        assert worker.get_latest_result() is None

    def test_start_stop_lifecycle(self):
        """Start()/stop() manage the background thread cleanly."""
        detector = Mock()
        detector.detect.return_value = _make_result()

        worker = AsyncInferenceWorker(detector)
        assert not worker.is_running

        worker.start()
        assert worker.is_running

        worker.stop(timeout=2.0)
        assert not worker.is_running

    def test_submit_frame_produces_result(self):
        """Submitting a frame eventually yields a detection result."""
        detector = Mock()
        detector.detect.return_value = _make_result()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        worker = AsyncInferenceWorker(detector)
        worker.start()
        try:
            worker.submit_frame(frame)

            deadline = time.time() + 2.0
            result = None
            while time.time() < deadline:
                result = worker.get_latest_result()
                if result is not None:
                    break
                time.sleep(0.01)

            assert result is not None
            detector.detect.assert_called()
        finally:
            worker.stop(timeout=2.0)

    def test_submit_frame_drops_oldest_when_queue_full(self):
        """Submitting frames beyond max_queue_size drops oldest, not blocks."""
        detector = Mock()
        detector.detect.return_value = _make_result()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        # Worker not started, so the queue never drains - exercises the
        # drop-oldest path directly.
        worker = AsyncInferenceWorker(detector, max_queue_size=1)
        worker.submit_frame(frame)
        worker.submit_frame(frame)  # Should not raise or block.
        worker.submit_frame(frame)

    def test_detector_error_does_not_kill_worker_thread(self):
        """An exception during detect() is swallowed and the worker keeps running."""
        detector = Mock()
        detector.detect.side_effect = [RuntimeError("boom"), _make_result()]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        worker = AsyncInferenceWorker(detector)
        worker.start()
        try:
            worker.submit_frame(frame)
            time.sleep(0.05)
            worker.submit_frame(frame)

            deadline = time.time() + 2.0
            result = None
            while time.time() < deadline:
                result = worker.get_latest_result()
                if result is not None:
                    break
                time.sleep(0.01)

            assert result is not None
            assert worker.is_running
        finally:
            worker.stop(timeout=2.0)
