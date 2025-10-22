"""Detector module for various object detection/segmentation backends."""

from .base_detector import BaseDetector, Detection, DetectionResult

__all__ = ["BaseDetector", "Detection", "DetectionResult"]
