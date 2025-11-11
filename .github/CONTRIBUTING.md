# Contributing to Tello Vision

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to Tello Vision. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Contributing to Tello Vision](#contributing-to-tello-vision)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [How Can I Contribute?](#how-can-i-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Your First Code Contribution](#your-first-code-contribution)
    - [Pull Requests](#pull-requests)
  - [Style Guides](#style-guides)
    - [Git Commit Messages](#git-commit-messages)
    - [Python Style Guide](#python-style-guide)
    - [Documentation Style Guide](#documentation-style-guide)
  - [Development Setup](#development-setup)
    - [1. Fork and Clone](#1-fork-and-clone)
    - [2. Create Virtual Environment](#2-create-virtual-environment)
    - [3. Install Development Dependencies](#3-install-development-dependencies)
    - [4. Create a Branch](#4-create-a-branch)
    - [5. Make Changes](#5-make-changes)
    - [6. Format and Lint](#6-format-and-lint)
    - [7. Test Your Changes](#7-test-your-changes)
    - [8. Commit and Push](#8-commit-and-push)
    - [9. Create Pull Request](#9-create-pull-request)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
  - [Adding a New Detector Backend](#adding-a-new-detector-backend)
    - [1. Create Detector Class](#1-create-detector-class)
    - [2. Register in Factory](#2-register-in-factory)
    - [3. Add Configuration](#3-add-configuration)
    - [4. Test It](#4-test-it)
    - [5. Update Documentation](#5-update-documentation)
  - [Additional Notes](#additional-notes)
    - [Issue and Pull Request Labels](#issue-and-pull-request-labels)
    - [Recognition](#recognition)
    - [Questions?](#questions)

## Code of Conduct

This project and everyone participating in it is governed by the [Tello Vision Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Configure with '...'
2. Run command '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or error logs.

**Environment:**

- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- PyTorch version: [e.g., 2.0.1]
- CUDA version (if using GPU): [e.g., 11.8]
- Detector backend: [YOLOv8/Detectron2]
- Tello firmware version: [if known]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most users
- **List any similar features** in other projects (if applicable)
- **Specify which version** you're using

**Areas we'd love contributions in:**

- New detector backends (RT-DETR, SAM, DINO, etc.)
- Object tracking integration (ByteTrack, DeepSORT, etc.)
- Path planning algorithms
- SLAM integration
- Additional autonomous behaviors
- Performance optimizations
- Documentation improvements
- Testing infrastructure
- CI/CD pipelines

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements
- `enhancement` - New features

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes** following our style guides
3. **Add tests** if you've added code that should be tested
4. **Update documentation** if you've changed APIs or added features
5. **Ensure the test suite passes** (when available)
6. **Make sure your code lints** (use `black` and `ruff`)
7. **Submit that pull request!**

**PR Template:**

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?

Describe the tests you ran and how to reproduce them.

## Checklist:

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
```

## Style Guides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - 🎨 `:art:` - Improve structure/format of code
  - ⚡ `:zap:` - Improve performance
  - 🐛 `:bug:` - Fix a bug
  - ✨ `:sparkles:` - Introduce new features
  - 📝 `:memo:` - Add or update documentation
  - 🚀 `:rocket:` - Deploy stuff
  - ✅ `:white_check_mark:` - Add or update tests
  - 🔧 `:wrench:` - Add or update configuration files
  - 🔨 `:hammer:` - Add or update development scripts
  - ♻️ `:recycle:` - Refactor code
  - 🎉 `:tada:` - Begin a project

**Examples:**

```
✨ Add YOLOv10 detector backend
🐛 Fix drone connection timeout on slow networks
📝 Update installation guide for Windows users
⚡ Optimize mask rendering performance
```

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Use `black` with default settings
- **Linter**: Use `ruff` for fast linting
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

**Example:**

```python
from typing import List, Optional
import numpy as np


class MyDetector(BaseDetector):
    """
    Custom detector implementation.

    This detector does X, Y, and Z using the ABC algorithm.

    Args:
        config: Configuration dictionary containing model parameters
        device: Device to run inference on ('cuda' or 'cpu')

    Attributes:
        model: The loaded detection model
        class_names: List of class names the model can detect
    """

    def __init__(self, config: dict, device: str = 'cuda'):
        """Initialize the detector with configuration."""
        super().__init__(config)
        self.device = device

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format

        Returns:
            DetectionResult containing all detections

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Detection logic here
        pass
```

**Code formatting commands:**

```bash
# Format code
black tello_vision/

# Lint code
ruff check tello_vision/

# Type check
mypy tello_vision/
```

### Documentation Style Guide

- Use **Markdown** for all documentation
- Use **code blocks** with language specification
- Include **examples** where possible
- Keep **line length** reasonable (80-100 chars)
- Use **headers** appropriately (h1 for title, h2 for sections, etc.)
- Include a **table of contents** for long documents
- Use **relative links** for internal documentation
- Add **alt text** for images

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/dronefreak/dji-tello-object-detection-segmentation.git
cd tello-vision
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev,yolo]"
```

This installs:

- The package in editable mode
- YOLOv8 backend
- Development tools (black, ruff, mypy, pytest)

### 4. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/issue-123
```

### 5. Make Changes

Edit files, add features, fix bugs, improve docs...

### 6. Format and Lint

```bash
# Format code
black tello_vision/

# Check linting
ruff check tello_vision/

# Fix auto-fixable issues
ruff check --fix tello_vision/

# Type check
mypy tello_vision/
```

### 7. Test Your Changes

```bash
# Test detector without drone
python examples/test_detector.py --source 0

# Run unit tests (when available)
pytest tests/

# Test with config
python -m tello_vision.app --config config.yaml
```

### 8. Commit and Push

```bash
git add .
git commit -m "✨ Add awesome new feature"
git push origin feature/my-new-feature
```

### 9. Create Pull Request

Go to GitHub and create a pull request from your branch.

## Project Structure

```
tello_vision/
├── tello_vision/           # Main package
│   ├── __init__.py
│   ├── app.py             # Main application
│   ├── tello_controller.py # Drone control
│   ├── visualizer.py      # Visualization
│   └── detectors/         # Detection backends
│       ├── __init__.py
│       ├── base_detector.py    # Abstract base class
│       ├── yolo_detector.py    # YOLOv8 implementation
│       └── detectron2_detector.py  # Detectron2 implementation
├── examples/              # Usage examples
│   ├── test_detector.py
│   ├── benchmark.py
│   └── object_follower.py
├── tests/                 # Unit tests (to be added)
├── docs/                  # Additional documentation
├── config.yaml            # Default configuration
├── pyproject.toml         # Package configuration
└── README.md             # Main documentation
```

**Key files to know:**

- `tello_vision/detectors/base_detector.py` - Add new detector backends here
- `tello_vision/tello_controller.py` - Modify drone behavior here
- `tello_vision/visualizer.py` - Customize visualization here
- `config.yaml` - Default configuration
- `examples/` - Reference implementations

## Testing

We're building out our testing infrastructure. Currently:

**Manual Testing:**

```bash
# Test detector
python examples/test_detector.py --source 0

# Benchmark
python examples/benchmark.py

# Full system test (requires drone)
python -m tello_vision.app
```

**Unit Tests (coming soon):**

```bash
pytest tests/
pytest tests/test_detectors.py -v
pytest --cov=tello_vision tests/
```

**What we need tests for:**

- Detector backends
- Visualization functions
- Configuration loading
- Drone controller (mocked)
- Data structures (Detection, DetectionResult)

## Adding a New Detector Backend

This is a common contribution. Here's how:

### 1. Create Detector Class

Create `tello_vision/detectors/my_detector.py`:

```python
from .base_detector import BaseDetector, Detection, DetectionResult
import time
import numpy as np

class MyDetector(BaseDetector):
    """My custom detector implementation."""

    def load_model(self) -> None:
        """Load the detection model."""
        # Load your model here
        self.model = load_my_model(self.config['model_path'])
        self.class_names = self.model.get_classes()
        self._initialized = True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on frame."""
        if not self._initialized:
            raise RuntimeError("Model not loaded")

        start = time.time()

        # Run your detection
        results = self.model.predict(frame)

        # Parse results into Detection objects
        detections = []
        for res in results:
            det = Detection(
                class_id=res.class_id,
                class_name=self.get_class_name(res.class_id),
                confidence=res.confidence,
                bbox=res.bbox,
                mask=res.mask if hasattr(res, 'mask') else None
            )
            detections.append(det)

        return DetectionResult(
            detections=detections,
            inference_time=time.time() - start,
            frame_shape=frame.shape
        )

    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        return self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
```

### 2. Register in Factory

Edit `tello_vision/detectors/base_detector.py`:

```python
@staticmethod
def create_detector(backend: str, config: dict) -> 'BaseDetector':
    if backend == 'yolov8':
        from .yolo_detector import YOLODetector
        return YOLODetector(config)
    elif backend == 'detectron2':
        from .detectron2_detector import Detectron2Detector
        return Detectron2Detector(config)
    elif backend == 'my_detector':  # Add this
        from .my_detector import MyDetector
        return MyDetector(config)
    else:
        raise ValueError(f"Unsupported detector backend: {backend}")
```

### 3. Add Configuration

Update `config.yaml`:

```yaml
detector:
  backend: "my_detector" # Can switch to this

  my_detector:
    model_path: "path/to/model"
    confidence: 0.5
    device: "cuda"
```

### 4. Test It

```bash
python examples/test_detector.py --source 0
```

### 5. Update Documentation

- Add to README.md
- Update CONTRIBUTING.md (this file)
- Add example usage

## Additional Notes

### Issue and Pull Request Labels

- `bug` - Something isn't working
- `documentation` - Improvements or additions to documentation
- `duplicate` - This issue or pull request already exists
- `enhancement` - New feature or request
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `invalid` - This doesn't seem right
- `question` - Further information is requested
- `wontfix` - This will not be worked on
- `performance` - Performance improvements
- `breaking-change` - Breaks backward compatibility

### Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Special thanks in documentation

### Questions?

Feel free to:

- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Reach out to maintainers

---
