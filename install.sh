#!/bin/bash
# Installation script for Tello Vision

set -e

echo "========================================="
echo "Tello Vision Installation"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python version OK: $python_version"
echo ""

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"
echo ""

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip wheel setuptools
echo "✓ pip upgraded"
echo ""

# Prompt for detector backend
echo "[3/4] Select detector backend:"
echo "  1) YOLOv8 (recommended - fast, real-time)"
echo "  2) Detectron2 (slower but more accurate)"
echo "  3) Both (requires more disk space)"
read -p "Enter choice [1-3]: " backend_choice

case $backend_choice in
    1)
        echo "Installing with YOLOv8..."
        pip install -e ".[yolo]"
        ;;
    2)
        echo "Installing with Detectron2..."
        # Install base dependencies first (includes torch)
        pip install -e .
        # Then install detectron2 (needs torch to build)
        pip install "git+https://github.com/facebookresearch/detectron2.git"
        ;;
    3)
        echo "Installing with both backends..."
        # Install base + yolo first (includes torch)
        pip install -e ".[yolo]"
        # Then install detectron2 (needs torch to build)
        echo "Installing Detectron2 (requires torch, installed above)..."
        pip install "git+https://github.com/facebookresearch/detectron2.git"
        ;;
    *)
        echo "Invalid choice. Installing YOLOv8 by default..."
        pip install -e ".[yolo]"
        ;;
esac

echo "✓ Dependencies installed"
echo ""

# Check for CUDA
echo "[4/4] Checking for CUDA..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✓ CUDA is available - GPU acceleration enabled"
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    echo "  CUDA version: $cuda_version"
else
    echo "⚠ CUDA not available - will use CPU (slower)"
    echo "  Consider installing CUDA for better performance"
fi
echo ""

# Create output directory
mkdir -p output
echo "✓ Created output directory"
echo ""

echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Quick start:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Power on your Tello and connect to its WiFi"
echo "  3. Run: python -m tello_vision.app"
echo ""
echo "Examples:"
echo "  - Test detector: python examples/test_detector.py --source 0"
echo "  - Benchmark: python examples/benchmark.py"
echo "  - Object following: python examples/object_follower.py"
echo ""
echo "Configuration:"
echo "  Edit config.yaml to customize settings"
echo ""