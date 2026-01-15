"""Tests for base detector functionality."""

import pytest

from tello_vision.detectors.base_detector import (
    BaseDetector,
    Detection,
    DetectionResult,
)


class TestDetection:
    """Tests for Detection class."""

    def test_detection_creation(self):
        """Test creating a detection."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 300, 400),
            mask=None,
        )

        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.85
        assert det.bbox == (100, 100, 300, 400)
        assert det.mask is None

    def test_detection_center(self):
        """Test calculating detection center."""
        det = Detection(
            class_id=0, class_name="person", confidence=0.85, bbox=(100, 100, 300, 400)
        )

        center = det.center
        assert center == (200, 250)  # ((100+300)/2, (100+400)/2)

    def test_detection_area(self):
        """Test calculating detection area."""
        det = Detection(
            class_id=0, class_name="person", confidence=0.85, bbox=(100, 100, 300, 400)
        )

        area = det.area
        assert area == 60000  # (300-100) * (400-100)


class TestDetectionResult:
    """Tests for DetectionResult class."""

    def test_detection_result_creation(self, mock_detection):
        """Test creating a detection result."""
        result = DetectionResult(
            detections=[mock_detection], inference_time=0.035, frame_shape=(480, 640, 3)
        )

        assert len(result.detections) == 1
        assert result.inference_time == 0.035
        assert result.frame_shape == (480, 640, 3)

    def test_detection_result_count(self, mock_detection):
        """Test detection count property."""
        result = DetectionResult(
            detections=[mock_detection, mock_detection],
            inference_time=0.035,
            frame_shape=(480, 640, 3),
        )

        assert result.count == 2

    def test_filter_by_class(self, mock_detection):
        """Test filtering detections by class."""
        det1 = Detection(0, "person", 0.9, (0, 0, 100, 100))
        det2 = Detection(1, "car", 0.8, (0, 0, 100, 100))
        det3 = Detection(0, "person", 0.7, (0, 0, 100, 100))

        result = DetectionResult([det1, det2, det3], 0.035, (480, 640, 3))

        filtered = result.filter_by_class(["person"])
        assert filtered.count == 2
        assert all(d.class_name == "person" for d in filtered.detections)

    def test_filter_by_confidence(self, mock_detection):
        """Test filtering detections by confidence."""
        det1 = Detection(0, "person", 0.9, (0, 0, 100, 100))
        det2 = Detection(1, "car", 0.5, (0, 0, 100, 100))
        det3 = Detection(2, "dog", 0.3, (0, 0, 100, 100))

        result = DetectionResult([det1, det2, det3], 0.035, (480, 640, 3))

        filtered = result.filter_by_confidence(0.6)
        assert filtered.count == 1
        assert filtered.detections[0].confidence == 0.9


class TestBaseDetector:
    """Tests for BaseDetector class."""

    def test_detector_factory_yolov8(self, sample_config):
        """Test creating YOLOv8 detector via factory."""
        detector = BaseDetector.create_detector(
            "yolov8", sample_config["detector"]["yolov8"]
        )

        assert detector is not None
        assert detector.device == "cpu"

    def test_detector_factory_invalid(self, sample_config):
        """Test factory with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported detector backend"):
            BaseDetector.create_detector("invalid_backend", {})

    def test_detector_not_initialized(self, sample_config):
        """Test detector before initialization."""
        detector = BaseDetector.create_detector(
            "yolov8", sample_config["detector"]["yolov8"]
        )

        assert not detector.is_initialized()

    def test_detector_warmup_fails_before_load(self, sample_config):
        """Test warmup fails before model is loaded."""
        detector = BaseDetector.create_detector(
            "yolov8", sample_config["detector"]["yolov8"]
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.warmup()


@pytest.mark.slow
@pytest.mark.integration
class TestYOLODetector:
    """Integration tests for YOLO detector."""

    def test_yolo_load_model(self, sample_config):
        """Test loading YOLO model."""
        pytest.importorskip("ultralytics")

        from tello_vision.detectors.yolo_detector import YOLODetector

        detector = YOLODetector(sample_config["detector"]["yolov8"])
        detector.load_model()

        assert detector.is_initialized()
        assert len(detector.class_names) > 0

    def test_yolo_detect(self, sample_config, sample_frame):
        """Test YOLO detection."""
        pytest.importorskip("ultralytics")

        from tello_vision.detectors.yolo_detector import YOLODetector

        detector = YOLODetector(sample_config["detector"]["yolov8"])
        detector.load_model()

        result = detector.detect(sample_frame)

        assert isinstance(result, DetectionResult)
        assert result.inference_time > 0
        assert result.frame_shape == sample_frame.shape

    def test_yolo_get_class_name(self, sample_config):
        """Test getting class names."""
        pytest.importorskip("ultralytics")

        from tello_vision.detectors.yolo_detector import YOLODetector

        detector = YOLODetector(sample_config["detector"]["yolov8"])
        detector.load_model()

        class_name = detector.get_class_name(0)
        assert isinstance(class_name, str)
        assert len(class_name) > 0
