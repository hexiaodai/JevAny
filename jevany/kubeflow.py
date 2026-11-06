"""Optional structured progress reporting for Kubeflow Trainer TrainJobs."""

import logging
import math
import os
import threading
import time


REPORT_INTERVAL = 5.0
logger = logging.getLogger(__name__)


class KubeflowStatus:
    """Publish the main process's latest metrics without blocking optimizer steps.

    The official SDK owns authentication, TLS, token rotation and HTTP errors.
    Only the latest snapshot is retained between reports; metric history remains
    the responsibility of the experiment tracker.
    """

    def __init__(self, total_steps: int, initial_step: int):
        self.total_steps = total_steps
        self.initial_step = initial_step
        self.started_at = time.monotonic()
        self.metrics = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = {}
        self._revision = 0
        self._reported = 0
        self._last_report = float("-inf")
        self._thread = None

    def __enter__(self):
        if not os.environ.get("KUBEFLOW_TRAINER_SERVER_URL"):
            return self
        try:
            from kubeflow.trainer import update_trainjob_status
        except ImportError:
            logger.warning("Kubeflow progress reporting requires JevAny's optional kubeflow extra in the training image")
            return self
        self._thread = threading.Thread(target=self._run, args=(update_trainjob_status,), daemon=True)
        self._thread.start()
        self.update(self.initial_step, {})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            # Allow one SDK throttle interval and one bounded HTTP request.
            self._thread.join(timeout=2 * REPORT_INTERVAL + 1)

    def update(self, step: int, metrics: dict[str, float | int | str]):
        """Use the real optimizer schedule, excluding work restored from a checkpoint."""
        if self._thread is None:
            return
        self.metrics.update(metrics)
        self.metrics.update(optimizer_step=step, total_steps=self.total_steps)
        snapshot = {
            "progress_percent": 100 * step // self.total_steps,
            "metrics": dict(self.metrics),
        }
        completed = step - self.initial_step
        if step == self.total_steps:
            snapshot["estimated_remaining_seconds"] = 0
        elif completed > 0:
            seconds_per_step = (time.monotonic() - self.started_at) / completed
            snapshot["estimated_remaining_seconds"] = min(
                math.ceil((self.total_steps - step) * seconds_per_step), 2**31 - 1,
            )
        with self._lock:
            self._snapshot = snapshot
            self._revision += 1

    def _run(self, report):
        while not self._stop.wait(REPORT_INTERVAL):
            self._send(report)
        # The SDK also throttles successful calls. Wait before the final report
        # so a short final interval does not discard the last optimizer step.
        with self._lock:
            pending = self._revision != self._reported
        if pending:
            delay = REPORT_INTERVAL - (time.monotonic() - self._last_report)
            if delay > 0:
                time.sleep(delay)
            self._send(report)

    def _send(self, report):
        with self._lock:
            if self._revision == self._reported:
                return
            snapshot, revision = self._snapshot, self._revision
        if report(**snapshot):
            self._last_report = time.monotonic()
            self._reported = revision
