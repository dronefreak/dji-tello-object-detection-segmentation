"""Benchmark different detector models.

Compares speed and optionally accuracy across different backends and model sizes.
"""

import argparse
import time
from typing import Dict

import numpy as np
import yaml

from tello_vision.detectors.base_detector import BaseDetector


def benchmark_detector(
    detector: BaseDetector, num_frames: int = 100, resolution: tuple = (960, 720)
) -> Dict:
    """Benchmark a detector.

    Args:
        detector: Detector instance
        num_frames: Number of frames to process
        resolution: Frame resolution (width, height)

    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking {detector.__class__.__name__}...")

    # Load model
    detector.load_model()

    # Warmup
    print("  Warming up...")
    detector.warmup(num_iterations=10)

    # Generate dummy frames
    frames = [
        np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    # Benchmark
    print(f"  Processing {num_frames} frames...")
    inference_times = []
    total_detections = 0

    start_time = time.time()

    for frame in frames:
        result = detector.detect(frame)
        inference_times.append(result.inference_time)
        total_detections += result.count

    total_time = time.time() - start_time

    # Calculate stats
    avg_inference = np.mean(inference_times)
    std_inference = np.std(inference_times)
    fps = num_frames / total_time

    return {
        "avg_inference_ms": avg_inference * 1000,
        "std_inference_ms": std_inference * 1000,
        "min_inference_ms": min(inference_times) * 1000,
        "max_inference_ms": max(inference_times) * 1000,
        "fps": fps,
        "total_time": total_time,
        "avg_detections": total_detections / num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark detectors")
    parser.add_argument(
        "--num-frames", type=int, default=100, help="Number of frames to process"
    )
    parser.add_argument(
        "--resolution", type=str, default="960x720", help="Frame resolution (WxH)"
    )
    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))
    resolution = (width, height)

    # Load base config
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Define models to benchmark
    benchmarks = [
        # YOLOv8 models
        (
            "YOLOv8n-seg (Nano)",
            "yolov8",
            {"model": "yolov8n-seg.pt", "device": "cuda", "confidence": 0.5},
        ),
        (
            "YOLOv8s-seg (Small)",
            "yolov8",
            {"model": "yolov8s-seg.pt", "device": "cuda", "confidence": 0.5},
        ),
        (
            "YOLOv8m-seg (Medium)",
            "yolov8",
            {"model": "yolov8m-seg.pt", "device": "cuda", "confidence": 0.5},
        ),
        # Detectron2
        ("Detectron2 R50-FPN", "detectron2", base_config["detector"]["detectron2"]),
    ]

    results = []

    print("=" * 80)
    print(
        f"Benchmarking Detectors - {args.num_frames}"
        f" frames at {resolution[0]}x{resolution[1]}"
    )
    print("=" * 80)
    print()

    for name, backend, config in benchmarks:
        try:
            detector = BaseDetector.create_detector(backend, config)
            result = benchmark_detector(detector, args.num_frames, resolution)
            result["name"] = name
            results.append(result)
            print("  ✓ Complete\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            continue

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    print(
        f"{'Model':<30} {'FPS':>8} {'Avg(ms)':>10}"
        f" {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
    )
    print("-" * 80)

    for result in sorted(results, key=lambda x: x["fps"], reverse=True):
        print(
            f"{result['name']:<30} "
            f"{result['fps']:>8.1f} "
            f"{result['avg_inference_ms']:>10.1f} "
            f"{result['std_inference_ms']:>10.1f} "
            f"{result['min_inference_ms']:>10.1f} "
            f"{result['max_inference_ms']:>10.1f}"
        )

    print()
    print("=" * 80)
    print("\nRecommendations:")

    fastest = max(results, key=lambda x: x["fps"])
    print(f"  Fastest: {fastest['name']} ({fastest['fps']:.1f} FPS)")

    most_stable = min(results, key=lambda x: x["std_inference_ms"])
    print(
        f"  Most Stable: {most_stable['name']}"
        f" (±{most_stable['std_inference_ms']:.1f}ms)"
    )

    print()


if __name__ == "__main__":
    main()
