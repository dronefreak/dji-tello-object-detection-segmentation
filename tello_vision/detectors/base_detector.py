"""Abstract base class for object detection/segmentation models.

Allows easy swapping between different backends (YOLOv8, Detectron2, custom).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """Single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None  # Binary mask if available

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class DetectionResult:
    """Complete detection result for a frame."""

    detections: List[Detection]
    inference_time: float  # seconds
    frame_shape: Tuple[int, int, int]  # (H, W, C)

    def filter_by_class(self, class_names: List[str]) -> "DetectionResult":
        """Filter detections by class names."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionResult(filtered, self.inference_time, self.frame_shape)

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(filtered, self.inference_time, self.frame_shape)

    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)


class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    def __init__(self, config: dict):
        """Initialize detector with configuration.

        Args:
            config: Dictionary containing detector configuration
        """
        self.config = config
        self.class_names: List[str] = []
        self._initialized = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format

        Returns:
            DetectionResult containing all detections
        """
        pass

    @abstractmethod
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        pass

    def warmup(self, num_iterations: int = 3) -> None:
        """Warmup the model with dummy input. Useful for GPU initialization.

        Args:
            num_iterations: Number of warmup iterations
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.detect(dummy_frame)

    def is_initialized(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self.config.get("device", "cpu")

    @staticmethod
    def create_detector(backend: str, config: dict) -> "BaseDetector":
        """Factory method to create detector instance.

        Args:
            backend: Detector backend name ('yolov8', 'detectron2', etc.)
            config: Configuration dictionary

        Returns:
            Detector instance

        Raises:
            ValueError: If backend is not supported
        """
        if backend == "yolov8":
            from .yolo_detector import YOLODetector

            return YOLODetector(config)
        elif backend == "detectron2":
            from .detectron2_detector import Detectron2Detector

            return Detectron2Detector(config)
        else:
            raise ValueError(f"Unsupported detector backend: {backend}")
