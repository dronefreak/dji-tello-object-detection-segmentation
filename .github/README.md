# Tello Vision v2.0 🚁🤖

Modern, modular instance segmentation and object detection for DJI Tello drones. Complete rewrite with SOTA models, clean architecture, and actual performance.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache-yellow.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

## 📖 Documentation Index

| Section                         | Description                           |
| ------------------------------- | ------------------------------------- |
| [Quickstart](QUICKSTART.md)     | Get up and running fast               |
| [Migration](MIGRATION.md)       | Notes on upgrading between versions   |
| [Improvements](IMPROVEMENTS.md) | Roadmap and ideas for future releases |
| [License](#license)             | License and legal details             |

## Features

- 🎯 **Multiple Detection Backends**: YOLOv8 (fast) or Detectron2 (accurate)
- 🚁 **Modern Drone Control**: Smooth RC controls, async processing
- 🎨 **Rich Visualization**: Masks, bboxes, labels with transparency
- 📹 **Recording & Photos**: Video recording and frame capture
- ⚙️ **Config-Driven**: YAML configuration for everything
- 🔌 **Pluggable Architecture**: Easy to add custom models
- 📊 **Real-time Stats**: FPS, battery, detection count

## Quick Start

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/dronefreak/dji-tello-object-detection-segmentation
cd dji-tello-object-detection-segmentation

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with YOLOv8 (recommended for real-time)
pip install -e ".[yolo]"

# OR install with Detectron2 (higher quality, slower)
# Detectron2 isn't a regular PyPI extra — install the base package first
# (for torch), then install Detectron2 manually from git:
pip install -e .
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# OR both YOLOv8 and Detectron2
pip install -e ".[yolo]"
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 2. Run It

```bash
# Test without a drone first (webcam)
python examples/test_detector.py --source 0

# With a Tello: power it on, connect to its WiFi, then
python -m tello_vision.app

# With custom config
python -m tello_vision.app --config my_config.yaml
```

See [QUICKSTART.md](QUICKSTART.md) for benchmarking, config tweaks, and
the autonomous-follow demo.

### Controls

| Action         | Key         |
| -------------- | ----------- |
| Takeoff        | Tab         |
| Land           | Backspace   |
| Emergency Stop | Esc         |
| Move           | W/A/S/D     |
| Up/Down        | Space/Shift |
| Rotate         | Q/E         |
| Record Video   | R           |
| Take Photo     | Enter       |
| Quit           | P           |

## Architecture

```text
tello_vision/
├── detectors/
│   ├── base_detector.py      # Abstract detector interface
│   ├── yolo_detector.py       # YOLOv8 implementation
│   └── detectron2_detector.py # Detectron2 implementation
├── tello_controller.py        # Drone control & video streaming
├── visualizer.py              # Detection visualization
└── app.py                     # Main application
```

### Adding Custom Models

Extend `BaseDetector`:

```python
from tello_vision.detectors.base_detector import BaseDetector, DetectionResult

class MyCustomDetector(BaseDetector):
    def load_model(self):
        # Load your model
        pass

    def detect(self, frame) -> DetectionResult:
        # Run inference
        pass
```

Register in `base_detector.py`:

```python
def create_detector(backend: str, config: dict):
    if backend == 'custom':
        from .my_custom_detector import MyCustomDetector
        return MyCustomDetector(config)
```

## Performance Comparison

| Model          | Device   | FPS   | mAP | Use Case        |
| -------------- | -------- | ----- | --- | --------------- |
| YOLOv8n-seg    | RTX 3060 | 25-30 | ~35 | Real-time, fast |
| YOLOv8s-seg    | RTX 3060 | 18-22 | ~38 | Balanced        |
| YOLOv8m-seg    | RTX 3060 | 12-15 | ~41 | Accuracy focus  |
| Detectron2 R50 | RTX 3060 | 8-12  | ~38 | High quality    |
| YOLOv8n-seg    | CPU      | 2-3   | ~35 | CPU fallback    |

_FPS measured at 960x720 resolution_

All settings (detection, visualization, performance, recording) live in
`config.yaml`, which is commented inline; see
[QUICKSTART.md](QUICKSTART.md#configuration-tweaks) for common tweaks.

## Advanced Usage

### Python API

```python
from tello_vision import TelloVisionApp

app = TelloVisionApp('config.yaml')
if app.initialize():
    app.run()
```

### Custom Processing Pipeline

```python
from tello_vision import TelloController, BaseDetector, Visualizer

# Initialize components
drone = TelloController(config)
detector = BaseDetector.create_detector('yolov8', config)
visualizer = Visualizer(config)

# Custom loop
drone.connect()
detector.load_model()

while True:
    frame = drone.get_frame()
    result = detector.detect(frame)

    # Custom logic here
    for det in result.detections:
        if det.class_name == 'person' and det.confidence > 0.8:
            print(f"Person detected at {det.center}")

    frame = visualizer.draw_detections(frame, result)
```

## Troubleshooting

### Connection Issues

```bash
# Check WiFi connection
ping 192.168.10.1

# Verify Tello firmware is up to date
# Check battery > 10%
```

### Performance Issues

- Use smaller model: `yolov8n-seg.pt` instead of `yolov8x-seg.pt`
- Enable frame skipping: `frame_skip: 1` or `2`
- Lower confidence threshold may increase speed
- Use GPU if available

### Import Errors

```bash
# YOLOv8
pip install ultralytics

# Detectron2 (Linux/Mac)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Detectron2 (Windows) - build from source or use pre-built wheels
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,yolo]"

# Install pre-commit hooks (runs all checks below automatically on commit)
pre-commit install

# Run all checks manually
pre-commit run --all-files

# Individual tools
ruff check tello_vision/ tests/ examples/
ruff format tello_vision/ tests/ examples/
mypy tello_vision/ tests/ examples/ --ignore-missing-imports
bandit -r tello_vision/ examples/ --confidence-level medium
```

## Roadmap

- [ ] Multi-drone support
- [ ] ROS2 integration
- [ ] Web interface
- [ ] Object tracking (ByteTrack)
- [ ] Autonomous navigation
- [ ] Dataset recording tool
- [ ] Model training pipeline
- [ ] Docker container

## License

Apache License, Version 2.0 - fork it, break it, make it better.

## Acknowledgments

- Original repo: [dronefreak/dji-tello-object-detection-segmentation](https://github.com/dronefreak/dji-tello-object-detection-segmentation)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [djitellopy](https://github.com/damiafuentes/DJITelloPy)

---
