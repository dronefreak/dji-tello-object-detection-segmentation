"""Tests for main application."""

from unittest.mock import Mock, patch

from tello_vision.app import TelloVisionApp


class TestTelloVisionApp:
    """Tests for TelloVisionApp class."""

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_app_creation(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test creating the application."""
        # Create temp config file
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        app = TelloVisionApp(str(config_path))

        assert app.config == sample_config
        assert not app.running
        assert app.frame_count == 0
        assert app.fps == 0.0

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_process_frame(
        self,
        mock_viz,
        mock_detector,
        mock_controller,
        sample_config,
        sample_frame,
        mock_detection_result,
        tmp_path,
    ):
        """Test processing a single frame."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        mock_detector.create_detector.return_value = mock_detector_instance

        mock_viz_instance = Mock()
        mock_viz_instance.draw_detections.return_value = sample_frame
        mock_viz_instance.draw_stats.return_value = sample_frame
        mock_viz_instance.draw_fps.return_value = sample_frame
        mock_viz.return_value = mock_viz_instance

        mock_controller_instance = Mock()
        mock_controller_instance.get_stats_text.return_value = ["Battery: 85%"]
        mock_controller_instance.is_recording = False
        mock_controller.return_value = mock_controller_instance

        app = TelloVisionApp(str(config_path))
        result_frame = app.process_frame(sample_frame)

        assert result_frame is not None
        mock_detector_instance.detect.assert_called_once()

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_process_frame_async_inference(
        self,
        mock_viz,
        mock_detector,
        mock_controller,
        sample_config,
        sample_frame,
        mock_detection_result,
        tmp_path,
    ):
        """process_frame() uses background worker when async_inference on."""
        import time

        import yaml

        sample_config["processing"]["async_inference"] = True
        sample_config["processing"]["max_queue_size"] = 1

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = mock_detection_result
        mock_detector.create_detector.return_value = mock_detector_instance

        mock_viz_instance = Mock()
        mock_viz_instance.draw_detections.return_value = sample_frame
        mock_viz_instance.draw_stats.return_value = sample_frame
        mock_viz_instance.draw_fps.return_value = sample_frame
        mock_viz.return_value = mock_viz_instance

        mock_controller_instance = Mock()
        mock_controller_instance.get_stats_text.return_value = ["Battery: 85%"]
        mock_controller_instance.is_recording = False
        mock_controller.return_value = mock_controller_instance

        app = TelloVisionApp(str(config_path))
        assert app.inference_worker is not None

        # No result yet: should still return a frame without blocking.
        result_frame = app.process_frame(sample_frame)
        assert result_frame is not None

        app.inference_worker.start()
        try:
            deadline = time.time() + 2.0
            while (
                time.time() < deadline
                and app.inference_worker.get_latest_result() is None
            ):
                app.process_frame(sample_frame)
                time.sleep(0.01)

            assert app.inference_worker.get_latest_result() is not None
        finally:
            app.inference_worker.stop(timeout=2.0)

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_update_fps(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test FPS counter update."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        app = TelloVisionApp(str(config_path))

        initial_fps = app.fps
        app.frame_count = 100

        import time

        app.last_fps_time = time.time() - 2.0  # 2 seconds ago

        app.update_fps()

        assert app.fps != initial_fps
        assert app.frame_count == 0  # Should reset

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    @patch("tello_vision.app.cv2.imwrite")
    def test_take_photo(
        self,
        mock_imwrite,
        mock_viz,
        mock_detector,
        mock_controller,
        sample_config,
        sample_frame,
        tmp_path,
    ):
        """Test taking a photo."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        app = TelloVisionApp(str(config_path))
        app.take_photo(sample_frame)

        mock_imwrite.assert_called_once()
        call_args = mock_imwrite.call_args[0]
        assert "tello_photo_" in call_args[0]

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_toggle_recording(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test toggling video recording."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        mock_controller_instance = Mock()
        mock_controller_instance.is_recording = False
        mock_controller.return_value = mock_controller_instance

        app = TelloVisionApp(str(config_path))

        # Toggle on
        app.toggle_recording()
        mock_controller_instance.start_recording.assert_called_once()

        # Toggle off
        mock_controller_instance.is_recording = True
        app.toggle_recording()
        mock_controller_instance.stop_recording.assert_called_once()

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_initialization(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test app initialization."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        mock_detector_instance = Mock()
        mock_detector.create_detector.return_value = mock_detector_instance

        mock_controller_instance = Mock()
        mock_controller_instance.connect.return_value = True
        mock_controller.return_value = mock_controller_instance

        app = TelloVisionApp(str(config_path))
        result = app.initialize()

        assert result is True
        mock_detector_instance.load_model.assert_called_once()
        mock_detector_instance.warmup.assert_called_once()
        mock_controller_instance.connect.assert_called_once()

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_initialization_failure(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test app initialization failure."""
        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        mock_controller_instance = Mock()
        mock_controller_instance.connect.return_value = False
        mock_controller.return_value = mock_controller_instance

        app = TelloVisionApp(str(config_path))
        result = app.initialize()

        assert result is False


class TestAppConfiguration:
    """Tests for configuration handling."""

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_output_directory_creation(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test output directory is created."""
        config_path = tmp_path / "test_config.yaml"
        sample_config["processing"]["output_dir"] = str(tmp_path / "custom_output")

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        app = TelloVisionApp(str(config_path))

        assert app.output_dir.exists()

    @patch("tello_vision.app.TelloController")
    @patch("tello_vision.app.BaseDetector")
    @patch("tello_vision.app.Visualizer")
    def test_detector_backend_selection(
        self, mock_viz, mock_detector, mock_controller, sample_config, tmp_path
    ):
        """Test correct detector backend is selected."""
        config_path = tmp_path / "test_config.yaml"

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        _ = TelloVisionApp(str(config_path))

        mock_detector.create_detector.assert_called_once_with(
            "yolov8", sample_config["detector"]["yolov8"]
        )
