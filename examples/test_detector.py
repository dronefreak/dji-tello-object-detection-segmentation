"""
Example: Basic detection without drone (using webcam or video file)
Useful for testing detector models without needing the actual drone.
"""

import argparse
import time

import cv2
import yaml

from tello_vision.detectors.base_detector import BaseDetector
from tello_vision.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 for webcam, or path to video file)",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize detector
    detector_config = config["detector"]
    backend = detector_config["backend"]
    detector = BaseDetector.create_detector(backend, detector_config[backend])

    print(f"Loading {backend} model...")
    detector.load_model()
    detector.warmup()

    # Initialize visualizer
    visualizer = Visualizer(config["visualization"])

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    print(f"Processing video from {args.source}")
    print("Press 'q' to quit")

    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            result = detector.detect(frame)

            # Visualize
            frame = visualizer.draw_detections(frame, result)

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()

            frame = visualizer.draw_fps(frame, fps)

            # Show stats
            stats = [
                f"Detections: {result.count}",
                f"Inference: {result.inference_time*1000:.1f}ms",
            ]
            frame = visualizer.draw_stats(frame, stats)

            # Display
            cv2.imshow("Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
