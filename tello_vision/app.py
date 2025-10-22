"""Main application for Tello Vision.

Integrates drone control, detection, and visualization.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .detectors.base_detector import BaseDetector
from .tello_controller import TelloController
from .visualizer import Visualizer


class TelloVisionApp:
    """Main application for Tello drone with vision capabilities."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize application.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.drone = TelloController(self.config["drone"])

        # Initialize detector
        detector_config = self.config["detector"]
        backend = detector_config["backend"]
        backend_config = detector_config[backend]

        self.detector = BaseDetector.create_detector(backend, backend_config)

        # Initialize visualizer
        self.visualizer = Visualizer(self.config["visualization"])

        # Processing config
        self.processing_config = self.config["processing"]

        # State
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

        # Output directory
        self.output_dir = Path(self.processing_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> bool:
        """Initialize all components.

        Returns:
            True if initialization successful
        """
        print("=" * 50)
        print("Tello Vision - Modern Instance Segmentation")
        print("=" * 50)

        # Load detector model
        print("\n[1/3] Loading detection model...")
        self.detector.load_model()
        self.detector.warmup()

        # Connect to drone
        print("\n[2/3] Connecting to drone...")
        if not self.drone.connect():
            print("Failed to connect to drone!")
            return False

        # Setup keyboard controls
        print("\n[3/3] Setting up controls...")
        self.drone.setup_keyboard_controls(self.config["controls"])

        print("\n✓ Initialization complete!")
        print("\nControls:")
        for action, key in self.config["controls"].items():
            print(f"  {action}: {key}")

        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and visualization.

        Args:
            frame: Input frame from drone

        Returns:
            Processed frame with visualizations
        """
        # Skip frames if configured
        frame_skip = self.processing_config.get("frame_skip", 0)
        if frame_skip > 0 and self.frame_count % (frame_skip + 1) != 0:
            return frame

        # Run detection
        result = self.detector.detect(frame)

        # Filter by target classes if configured
        target_classes = self.config["detector"].get("target_classes", [])
        if target_classes:
            result = result.filter_by_class(target_classes)

        # Draw detections
        frame = self.visualizer.draw_detections(frame, result)

        # Draw stats if enabled
        if self.processing_config.get("display_stats", True):
            stats = self.drone.get_stats_text()
            stats.append(f"Detections: {result.count}")
            stats.append(f"Inference: {result.inference_time*1000:.1f}ms")
            frame = self.visualizer.draw_stats(frame, stats)

        # Draw FPS if enabled
        if self.processing_config.get("display_fps", True):
            frame = self.visualizer.draw_fps(frame, self.fps)

        # Record if enabled
        if self.drone.is_recording:
            self.drone.write_frame(frame)

        return frame

    def update_fps(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

            # Update drone stats
            self.drone.update_stats()

    def run(self) -> None:
        """Run the main application loop."""
        self.running = True

        print("\n" + "=" * 50)
        print("Starting video stream...")
        print("Press 'tab' to takeoff, 'backspace' to land")
        print("Press 'p' to quit")
        print("=" * 50 + "\n")

        window_name = self.processing_config.get("window_name", "Tello Vision")

        try:
            while self.running:
                # Get frame from drone
                frame = self.drone.get_frame()

                if frame is None:
                    continue

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display if enabled
                if self.processing_config.get("display_window", True):
                    cv2.imshow(window_name, processed_frame)

                # Update FPS
                self.update_fps()

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p") or key == 27:  # 'p' or ESC
                    break
                elif key == ord("r"):
                    self.toggle_recording()
                elif key == 13:  # Enter
                    self.take_photo(processed_frame)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.shutdown()

    def toggle_recording(self) -> None:
        """Toggle video recording."""
        if not self.drone.is_recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"tello_video_{timestamp}.mp4")
            self.drone.start_recording(output_path)
        else:
            self.drone.stop_recording()

    def take_photo(self, frame: np.ndarray) -> None:
        """Save current frame as photo."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(self.output_dir / f"tello_photo_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Photo saved: {output_path}")

    def shutdown(self) -> None:
        """Shutdown application and cleanup."""
        print("\nShutting down...")
        self.running = False

        # Cleanup
        cv2.destroyAllWindows()
        self.drone.disconnect()

        print("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tello Vision - Modern instance segmentation for DJI Tello"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--no-drone",
        action="store_true",
        help="Run without connecting to drone (test mode)",
    )

    args = parser.parse_args()

    # Create and run app
    app = TelloVisionApp(args.config)

    if app.initialize():
        app.run()


if __name__ == "__main__":
    main()
