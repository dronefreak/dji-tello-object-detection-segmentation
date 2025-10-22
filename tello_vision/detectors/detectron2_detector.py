"""Detectron2 detector implementation.

Higher quality but slower than YOLO. Good for precision applications.
"""

import time

import numpy as np

from .base_detector import BaseDetector, Detection, DetectionResult


class Detectron2Detector(BaseDetector):
    """Detectron2 Mask R-CNN detector."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.predictor = None
        self.metadata = None

    def load_model(self) -> None:
        """Load Detectron2 model."""
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2.engine import DefaultPredictor
        except ImportError:
            raise ImportError(
                "detectron2 not installed. Install from: "
                "https://github.com/facebookresearch/detectron2"
            )

        cfg = get_cfg()

        # Load config
        config_file = self.config.get(
            "config_file", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.merge_from_file(model_zoo.get_config_file(config_file))

        # Set model weights
        weights = self.config.get("model_weights")
        if weights and weights.startswith("detectron2://"):
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        else:
            cfg.MODEL.WEIGHTS = weights or model_zoo.get_checkpoint_url(config_file)

        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.get("confidence", 0.5)

        # Set device
        device = self.config.get("device", "cuda")
        cfg.MODEL.DEVICE = device

        print(f"Loading Detectron2 model: {config_file} on {device}")

        # Create predictor
        self.predictor = DefaultPredictor(cfg)

        # Get metadata for class names
        dataset_name = config_file.split("/")[0]
        if dataset_name.startswith("COCO"):
            self.metadata = MetadataCatalog.get("coco_2017_val")
        else:
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        self.class_names = self.metadata.thing_classes

        self._initialized = True
        print(f"Detectron2 model loaded. Classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run Detectron2 detection on frame.

        Args:
            frame: Input image (H, W, C) in BGR format

        Returns:
            DetectionResult with all detections
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Run inference
        outputs = self.predictor(frame)

        inference_time = time.time() - start_time

        # Parse results
        detections = []
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            # Get masks if available
            masks = None
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()

            for idx in range(len(instances)):
                bbox = boxes[idx].astype(int)

                # Get mask
                mask = None
                if masks is not None:
                    mask = masks[idx].astype(np.uint8)

                detection = Detection(
                    class_id=int(classes[idx]),
                    class_name=self.get_class_name(int(classes[idx])),
                    confidence=float(scores[idx]),
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
            "backend": "detectron2",
            "config": self.config.get("config_file", "unknown"),
            "device": self.device,
            "num_classes": len(self.class_names),
            "classes": self.class_names,
        }
