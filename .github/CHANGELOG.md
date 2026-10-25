### Tello Vision – Changelog Summary

**Unreleased**

- **Safety**: keyboard controls no longer fire when the video window
  isn't focused. Takeoff/land/emergency are read from the OpenCV window
  only, and held movement keys (still tracked with pynput) only move the
  drone once the window has also seen that key. Previously pynput
  captured keys system-wide, so e.g. pressing Tab in another app took off.
- **Breaking**: the default "down" key changed from Shift to X. Modifier
  keys never reach the OpenCV window on their own, so they can't be
  focus-checked; such keys are now disabled with a startup warning.
- **Safety**: Esc no longer quits the main app. It is the emergency motor
  stop, so quitting with it in flight dropped the drone. Quit with P
  (lands first).
- **Safety**: takeoff/land/emergency failures (e.g. low battery, no
  response) are reported instead of raised. An exception in a pynput
  callback used to stop the listener, taking the emergency key with it.
  `takeoff()`/`land()` now return a success flag; a failed land keeps
  the drone marked as flying so shutdown retries landing.
- **Safety**: takeoff/land run on a background thread, so the video loop
  and emergency key stay responsive during a slow takeoff (up to ~20s).
- **Fix**: object follower now sends a hover command when the target is
  lost. The Tello keeps flying at its last RC velocity, so the drone
  previously kept moving.
- **Fix**: object follower smoothing modified detections in place, which
  fed smoothed values back into the average (adding lag) and drew the
  shifted box instead of the real detection.
- **Fix**: object follower PIDs are reset when auto-follow is enabled,
  avoiding a jolt from error accumulated while on the ground.
- **Fix**: `TelloController.get_frame()` returns a copy. djitellopy keeps
  returning the same array until a new frame arrives, so overlays stacked
  up across loop iterations and were fed to the detector. The async
  inference worker also gets its own copy of each frame.
- **Fix**: `--no-drone` mode no longer polls drone stats every second
  (which only printed errors).
- **Config**: default `processing.max_queue_size` lowered from 3 to 1;
  each extra slot added a full inference time of detection delay.
- **Tooling**: PR size hard limit raised from 800 to 1000 LOC.

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
