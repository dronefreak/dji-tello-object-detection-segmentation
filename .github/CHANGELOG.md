### Tello Vision – Changelog Summary

**Unreleased**

- **Fix**: `tello_vision` could not be imported on displayless systems
  (e.g. CI runners) - `pynput.keyboard` performs X11 backend detection
  at import time, which raised `ImportError` with no `DISPLAY`. The
  import is now deferred to where it's actually needed.
- **Fix**: CI's Documentation Check job checked for
  `README.md`/`CODE_OF_CONDUCT.md`/`SECURITY.md` at the repo root; these
  live under `.github/`.
- **Tooling**: mypy added as a blocking pre-commit hook and CI check
  (`tello_vision/`, `tests/`, `examples/`).
- **Tooling**: bandit added as a blocking pre-commit hook and CI check
  (medium confidence, `tello_vision/`, `examples/`).
- **Fix**: the `# noqa: S311 nosec B311` suppression comments in
  `visualizer.py` didn't actually suppress bandit's B311 finding -
  bandit's nosec regex requires `nosec` to immediately follow a `#`.
  Reordered to `# nosec B311  # noqa: S311`, which both tools honor.
- **Tooling**: yamllint, markdownlint, shellcheck/shfmt, and
  conventional-commit (commit-msg) hooks added to pre-commit. Fixed the
  violations they surfaced: broken doc-index links and a missing
  code-fence language in `.github/README.md`, and a shellcheck/shfmt
  cleanup of `install.sh` (quoted `read`, a `# shellcheck disable`
  for the not-yet-created venv activate script, consistent 4-space
  indentation).

**Version 2.0.0 – 2025-01-15**
A complete rewrite of the codebase with major modernization, performance improvements, and extensibility enhancements.

**Highlights**:

- **Architecture & Design**: Modular, pluggable detector backends; abstract `BaseDetector` for custom models.
- **Detectors**: YOLOv8 (fast, default) and Detectron2 (high accuracy) support.
- **Drone Control**: Modern `TelloController` with djitellopy, RC support, and async frame processing.
- **Visualization**: Rich overlays with masks, bounding boxes, labels, and transparency.
- **Configuration**: YAML-based, replacing hardcoded values.
- **Performance**: 5–6x faster inference, 50% less memory usage, 6x faster model loading.
- **Testing & CI**: 60+ tests, pytest markers, GitHub Actions pipeline, pre-commit hooks.
- **Documentation**: Comprehensive guides, migration instructions, and example scripts for detection, tracking, and benchmarking.
- **Security**: Policies, bandit scanning, dependency safety checks, and pre-commit private key detection.

**Breaking Changes**:

- API incompatible with v1.x.
- Python 3.10+ required; TensorFlow replaced by PyTorch/Ultralytics.
- Legacy Mask R-CNN and TelloPy removed.

**Improvements**:

- Cleaner, maintainable, and type-hinted codebase.
- Extensible architecture for custom models.
- Real-time FPS counter, video/photo capture, and autonomous object tracking.
- Automated installation and setup scripts.

**Version 1.0.0 – 2019 (Legacy)**

- Initial implementation using Mask R-CNN, Tello control, and basic visualization.
- Known issues: slow performance, outdated dependencies, no tests, monolithic structure, and limited extensibility.

**Migration**: See [MIGRATION.md](MIGRATION.md) for detailed steps from v1.x to v2.0.

**Versioning**: Follows [SemVer](http://semver.org/): MAJOR for breaking changes, MINOR for new features, PATCH for bug fixes.

**Repository & Contributions**:

- [GitHub Repo](https://github.com/dronefreak/dji-tello-object-detection-segmentation)
- [Issues](https://github.com/dronefreak/dji-tello-object-detection-segmentation/issues)
- Original repository: [dronefreak/dji-tello-object-detection-segmentation](https://github.com/dronefreak/dji-tello-object-detection-segmentation)
