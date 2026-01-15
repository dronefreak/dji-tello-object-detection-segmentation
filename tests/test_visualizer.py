"""Tests for visualization functionality."""

import numpy as np

from tello_vision.detectors.base_detector import Detection
from tello_vision.visualizer import Visualizer


class TestVisualizer:
    """Tests for Visualizer class."""

    def test_visualizer_creation(self, sample_config):
        """Test creating a visualizer."""
        viz = Visualizer(sample_config["visualization"])

        assert viz.config == sample_config["visualization"]
        assert isinstance(viz.class_colors, dict)

    def test_get_color_consistent(self, sample_config):
        """Test color generation is consistent."""
        viz = Visualizer(sample_config["visualization"])

        color1 = viz.get_color("person")
        color2 = viz.get_color("person")

        assert color1 == color2
        assert len(color1) == 3
        assert all(0 <= c <= 255 for c in color1)

    def test_get_color_different_classes(self, sample_config):
        """Test different classes get different colors."""
        viz = Visualizer(sample_config["visualization"])

        color1 = viz.get_color("person")
        color2 = viz.get_color("car")

        assert color1 != color2

    def test_draw_detection(self, sample_config, sample_frame, mock_detection):
        """Test drawing a single detection."""
        viz = Visualizer(sample_config["visualization"])

        result_frame = viz.draw_detection(sample_frame.copy(), mock_detection)

        assert result_frame.shape == sample_frame.shape
        assert not np.array_equal(result_frame, sample_frame)  # Should be modified

    def test_draw_detections(self, sample_config, sample_frame, mock_detection_result):
        """Test drawing multiple detections."""
        viz = Visualizer(sample_config["visualization"])

        result_frame = viz.draw_detections(sample_frame.copy(), mock_detection_result)

        assert result_frame.shape == sample_frame.shape

    def test_draw_detection_no_mask(self, sample_config, sample_frame):
        """Test drawing detection without mask."""
        viz = Visualizer(sample_config["visualization"])

        det = Detection(0, "person", 0.85, (100, 100, 300, 400), mask=None)

        result_frame = viz.draw_detection(sample_frame.copy(), det)
        assert result_frame.shape == sample_frame.shape

    def test_draw_stats(self, sample_config, sample_frame):
        """Test drawing statistics."""
        viz = Visualizer(sample_config["visualization"])

        stats = ["Battery: 85%", "FPS: 25.5", "Detections: 3"]
        result_frame = viz.draw_stats(sample_frame.copy(), stats)

        assert result_frame.shape == sample_frame.shape
        assert not np.array_equal(result_frame, sample_frame)

    def test_draw_fps(self, sample_config, sample_frame):
        """Test drawing FPS counter."""
        viz = Visualizer(sample_config["visualization"])

        result_frame = viz.draw_fps(sample_frame.copy(), 25.5)

        assert result_frame.shape == sample_frame.shape

    def test_draw_crosshair(self, sample_config, sample_frame):
        """Test drawing crosshair."""
        viz = Visualizer(sample_config["visualization"])

        result_frame = viz.draw_crosshair(sample_frame.copy())

        assert result_frame.shape == sample_frame.shape
        assert not np.array_equal(result_frame, sample_frame)

    def test_draw_label(self, sample_config, sample_frame):
        """Test drawing label with background."""
        viz = Visualizer(sample_config["visualization"])

        frame = sample_frame.copy()
        viz._draw_label(frame, "Test Label", (50, 50), (255, 0, 0))

        assert not np.array_equal(frame, sample_frame)

    def test_draw_mask_with_alpha(self, sample_config, sample_frame):
        """Test drawing mask with transparency."""
        viz = Visualizer(sample_config["visualization"])

        mask = np.zeros((sample_frame.shape[0], sample_frame.shape[1]), dtype=np.uint8)
        mask[100:200, 100:200] = 1

        result_frame = viz._draw_mask(sample_frame.copy(), mask, (255, 0, 0))

        assert result_frame.shape == sample_frame.shape

    def test_visualization_config_options(
        self, sample_config, sample_frame, mock_detection
    ):
        """Test different visualization config options."""
        # Test with boxes only
        config = sample_config["visualization"].copy()
        config["show_masks"] = False
        config["show_labels"] = False

        viz = Visualizer(config)
        result = viz.draw_detection(sample_frame.copy(), mock_detection)
        assert result.shape == sample_frame.shape

        # Test with masks only
        config["show_boxes"] = False
        config["show_masks"] = True

        viz = Visualizer(config)
        result = viz.draw_detection(sample_frame.copy(), mock_detection)
        assert result.shape == sample_frame.shape
