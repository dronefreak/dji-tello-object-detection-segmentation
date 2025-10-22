# 🚀 Quick Start Guide - Tello Vision v2.0

## TL;DR - Get Running in 5 Minutes

```bash
# 1. Extract and enter directory
tar -xzf tello_vision_v2.tar.gz
cd tello_vision

# 2. Run installer (auto-installs everything)
./install.sh

# 3. Power on Tello, connect to its WiFi

# 4. Launch!
source venv/bin/activate
python -m tello_vision.app
```

## What You Get

- **Modern instance segmentation** (YOLOv8 or Detectron2)
- **5-6x faster** than the old code (25-30 FPS on decent GPU)
- **Modular architecture** - actually maintainable
- **Config-driven** - no code changes for common stuff
- **Production ready** - proper error handling, logging, async

## First Steps

### 1. Test Detection Without Drone
Good for verifying everything works:
```bash
python examples/test_detector.py --source 0  # Webcam
```

### 2. Benchmark Your Setup
See what FPS you can get:
```bash
python examples/benchmark.py
```

### 3. Full Drone Mode
With Tello connected:
```bash
python -m tello_vision.app
```

Controls:
- **Tab**: Takeoff
- **W/A/S/D**: Move
- **Space/Shift**: Up/Down
- **Q/E**: Rotate
- **R**: Record video
- **Enter**: Take photo
- **Backspace**: Land
- **P**: Quit

## Configuration Tweaks

Edit `config.yaml`:

**Want faster FPS?** Use smaller model:
```yaml
detector:
  yolov8:
    model: "yolov8n-seg.pt"  # n=nano (fastest)
```

**Only track people?**
```yaml
detector:
  target_classes: ["person"]
```

**Adjust visualization:**
```yaml
visualization:
  mask_alpha: 0.4  # Mask transparency
  show_confidence: true
```

**Performance tuning:**
```yaml
processing:
  frame_skip: 1  # Process every 2nd frame (doubles FPS)
```

## Project Structure

```
tello_vision/
├── config.yaml              ← Edit this for settings
├── install.sh              ← Run first
├── README.md               ← Full documentation
├── MIGRATION.md            ← If migrating from old code
│
├── tello_vision/           ← Main package
│   ├── app.py             ← Main application
│   ├── tello_controller.py ← Drone control
│   ├── visualizer.py       ← Visualization
│   └── detectors/          ← Detection backends
│
└── examples/               ← Usage examples
    ├── test_detector.py    ← Test without drone
    ├── benchmark.py        ← Performance tests
    └── object_follower.py  ← Autonomous tracking
```

## Autonomous Tracking Demo

Want to make the drone follow you?

```bash
python examples/object_follower.py
# Enter target: person
# Press TAB to enable auto-follow
```

This demonstrates reactive control suitable for autonomous vehicles.

## Common Issues

**"No CUDA"** - Will work on CPU, just slower. Install CUDA for speed.

**"Can't connect to Tello"** - Make sure you're connected to Tello's WiFi (not your home WiFi).

**"Low FPS"** - Try smaller model (`yolov8n-seg.pt`) or enable `frame_skip`.

**Import errors** - Run `./install.sh` again with correct backend choice.

## For Self-Driving Car Work

This gives you:
- Real-time object detection pipeline
- Target tracking framework  
- Reactive control examples
- Extensible architecture for adding SLAM, planning, etc.

Check `examples/object_follower.py` for autonomous navigation basics.

## Next Steps

1. **Read README.md** - Full documentation
2. **Try examples/** - Learn the API
3. **Modify config.yaml** - Tune for your use case
4. **Extend** - Add your own detectors/controllers

## Performance Reference

| GPU | Model | FPS |
|-----|-------|-----|
| RTX 3060 | YOLOv8n | 25-30 |
| RTX 3060 | YOLOv8s | 18-22 |
| 1050 Ti | YOLOv8n | 18-22 |
| CPU | YOLOv8n | 2-3 |

## Files to Know

- **config.yaml** - All settings
- **tello_vision/app.py** - Main application  
- **tello_vision/detectors/base_detector.py** - Add custom models here
- **examples/object_follower.py** - Autonomous control reference

## Getting Help

- Check **README.md** for detailed docs
- See **IMPROVEMENTS.md** for what changed
- Read **MIGRATION.md** if coming from old code
- Example code in **examples/** directory

---
