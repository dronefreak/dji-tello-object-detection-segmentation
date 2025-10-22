"""YOLOv8 detector implementation using Ultralytics.

Fast, real-time capable, and easy to use.
"""

import time

import cv2
import numpy as np

from .base_detector import BaseDetector, Detection, DetectionResult


class YOLODetector(BaseDetector):
    """YOLOv8 instance segmentation detector."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None

    def load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        model_name = self.config.get("model", "yolov8n-seg.pt")
        device = self.config.get("device", "cuda")

        print(f"Loading YOLOv8 model: {model_name} on {device}")
        self.model = YOLO(model_name)

        # Move to device
        self.model.to(device)

        # Get class names
        self.class_names = list(self.model.names.values())

        self._initialized = True
        print(f"YOLOv8 model loaded. Classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run YOLOv8 detection on frame.

        Args:
            frame: Input image (H, W, C) in BGR format

        Returns:
            DetectionResult with all detections
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Run inference
        results = self.model(
            frame,
            conf=self.config.get("confidence", 0.5),
            iou=self.config.get("iou_threshold", 0.45),
            verbose=False,
        )[0]

        inference_time = time.time() - start_time

        # Parse results
        detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Get masks if available
            masks = None
            if hasattr(results, "masks") and results.masks is not None:
                masks = results.masks.data.cpu().numpy()

            for idx in range(len(boxes)):
                class_id = class_ids[idx]
                bbox = boxes[idx].astype(int)

                # Get mask if available
                mask = None
                if masks is not None and idx < len(masks):
                    # Resize mask to original frame size
                    mask_resized = cv2.resize(
                        masks[idx],
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask = (mask_resized > 0.5).astype(np.uint8)

                detection = Detection(
                    class_id=class_id,
                    class_name=self.get_class_name(class_id),
                    confidence=float(confidences[idx]),
                    bbox=tuple(bbox),
                    mask=mask,
                )
                detections.append(detection)

        return DetectionResult(
            detections=detections,
            inference_time=inference_time,
            frame_shape=frame.shape,
        )

    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "backend": "yolov8",
            "model": self.config.get("model", "unknown"),
            "device": self.device,
            "num_classes": len(self.class_names),
            "classes": self.class_names,
        }
