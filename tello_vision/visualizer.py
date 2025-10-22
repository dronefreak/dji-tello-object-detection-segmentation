"""Visualization utilities for rendering detection results on frames."""

import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .detectors.base_detector import Detection, DetectionResult


class Visualizer:
    """Visualize detection results on frames."""

    def __init__(self, config: dict):
        """Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}

        # Load predefined colors if available
        if "class_colors" in config:
            self.class_colors = {
                name: tuple(color) for name, color in config["class_colors"].items()
            }

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a class.

        Args:
            class_name: Name of the class

        Returns:
            RGB color tuple
        """
        if class_name not in self.class_colors:
            # Generate random but consistent color
            random.seed(hash(class_name))
            self.class_colors[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return self.class_colors[class_name]

    def draw_detection(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Draw a single detection on the frame.

        Args:
            frame: Input frame
            detection: Detection to draw

        Returns:
            Frame with detection drawn
        """
        color = self.get_color(detection.class_name)
        x1, y1, x2, y2 = detection.bbox

        # Draw mask if available and enabled
        if self.config.get("show_masks", True) and detection.mask is not None:
            frame = self._draw_mask(frame, detection.mask, color)

        # Draw bounding box if enabled
        if self.config.get("show_boxes", True):
            thickness = self.config.get("box_thickness", 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label if enabled
        if self.config.get("show_labels", True):
            label = detection.class_name

            if self.config.get("show_confidence", True):
                label = f"{label} {detection.confidence:.2f}"

            self._draw_label(frame, label, (x1, y1), color)

        return frame

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw all detections on the frame.

        Args:
            frame: Input frame
            result: Detection results

        Returns:
            Frame with all detections drawn
        """
        for detection in result.detections:
            frame = self.draw_detection(frame, detection)

        return frame

    def _draw_mask(
        self, frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw segmentation mask with transparency."""
        alpha = self.config.get("mask_alpha", 0.4)

        # Create colored mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color

        # Blend with original frame
        frame = cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)

        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.get("font_scale", 0.6)
        thickness = self.config.get("font_thickness", 2)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width + 5, y),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 2, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def draw_stats(
        self, frame: np.ndarray, stats: List[str], position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """Draw statistics text on frame.

        Args:
            frame: Input frame
            stats: List of stat strings to display
            position: Starting position (x, y)

        Returns:
            Frame with stats drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 255, 0)
        line_height = 30

        x, y = position

        for i, stat in enumerate(stats):
            cv2.putText(
                frame,
                stat,
                (x, y + i * line_height),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_fps(
        self, frame: np.ndarray, fps: float, position: Tuple[int, int] = None
    ) -> np.ndarray:
        """Draw FPS counter on frame.

        Args:
            frame: Input frame
            fps: Current FPS
            position: Position to draw (default: top-right)

        Returns:
            Frame with FPS drawn
        """
        if position is None:
            position = (frame.shape[1] - 150, 30)

        text = f"FPS: {fps:.1f}"

        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return frame

    def draw_crosshair(
        self,
        frame: np.ndarray,
        size: int = 20,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw crosshair at center of frame.

        Args:
            frame: Input frame
            size: Size of crosshair
            color: Color of crosshair

        Returns:
            Frame with crosshair
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Draw horizontal line
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)

        # Draw vertical line
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)

        # Draw center circle
        cv2.circle(frame, (cx, cy), 5, color, -1)

        return frame
