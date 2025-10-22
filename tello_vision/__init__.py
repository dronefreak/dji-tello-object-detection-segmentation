"""Tello Vision - Modern instance segmentation for DJI Tello drones."""

__version__ = "2.0.0"

from .app import TelloVisionApp
from .detectors import BaseDetector, Detection, DetectionResult
from .tello_controller import TelloController
from .visualizer import Visualizer

__all__ = [
    "TelloVisionApp",
    "TelloController",
    "Visualizer",
    "BaseDetector",
    "Detection",
    "DetectionResult",
]
