"""Asynchronous inference worker.

Decouples frame capture/display from model inference by running detection on a dedicated
background thread. The main loop can keep grabbing and displaying frames at full speed
while inference proceeds independently; if inference is slower than capture, older un-
processed frames are dropped in favor of the newest one so results stay as fresh as
possible instead of the app falling behind a growing backlog.
"""

import queue
import threading
from typing import Optional

import numpy as np

from .detectors.base_detector import BaseDetector, DetectionResult

# How long the worker waits for a new frame before re-checking the stop
# event, so shutdown() doesn't hang if no frames are being submitted.
QUEUE_POLL_TIMEOUT = 0.1


class AsyncInferenceWorker:
    """Runs detector inference on a background thread.

    Frames are submitted via `submit_frame()`. If the internal queue is full (the worker
    hasn't kept up), the oldest queued frame is dropped in favor of the newest one,
    bounding latency instead of letting inference fall behind a stale backlog. The
    latest completed result is available via `get_latest_result()`.
    """

    def __init__(self, detector: BaseDetector, max_queue_size: int = 1):
        """Initialize the worker.

        Args:
            detector: Detector instance to run inference with. Must already
                be loaded (load_model()/warmup() called) before start().
            max_queue_size: Maximum number of frames buffered for
                inference before older frames start being dropped.
        """
        self.detector = detector
        self._input_queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=max(1, max_queue_size)
        )
        self._result_lock = threading.Lock()
        self._latest_result: Optional[DetectionResult] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Whether the background inference thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background inference thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="AsyncInferenceWorker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and wait for it to exit.

        Args:
            timeout: Max seconds to wait for the thread to finish.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit a frame for inference, dropping the oldest if the queue is full.

        Args:
            frame: Frame to run detection on.
        """
        try:
            self._input_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest queued frame in favor of the newest one, to
            # keep inference results as fresh as possible.
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait(frame)
            except queue.Full:
                pass  # Another thread raced us; safe to drop this frame.

    def get_latest_result(self) -> Optional[DetectionResult]:
        """Get the most recently completed detection result, if any.

        Returns:
            The latest DetectionResult, or None if inference hasn't
            produced a result yet (e.g. still warming up).
        """
        with self._result_lock:
            return self._latest_result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._input_queue.get(timeout=QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue

            try:
                result = self.detector.detect(frame)
            except Exception as e:  # noqa: BLE001 - keep worker alive on error
                print(f"Async inference error: {e}")
                continue

            with self._result_lock:
                self._latest_result = result
