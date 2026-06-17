"""Background memory monitor using psutil.

Samples both process RSS and system-wide used memory on a configurable
interval, tracking the peak seen during a benchmark run.

Usage::

    monitor = MemoryMonitor(interval_s=0.25)
    monitor.start()
    # ... run inference ...
    monitor.stop()
    print(monitor.peak_process_rss_mb, monitor.peak_system_used_mb)
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import psutil

_DEFAULT_INTERVAL_S = 0.25


class MemoryMonitor:
    """Polls psutil on a background thread to find peak RSS and system memory.

    Parameters
    ----------
    interval_s:
        Sampling period in seconds.
    sampler:
        Optional callable ``() -> tuple[float, float]`` returning
        ``(process_rss_bytes, system_used_bytes)``.  Injected in tests to avoid
        real psutil calls.
    """

    def __init__(
        self,
        interval_s: float = _DEFAULT_INTERVAL_S,
        *,
        sampler: Callable[[], tuple[float, float]] | None = None,
    ) -> None:
        self._interval_s = interval_s
        self._sampler = sampler or self._default_sampler
        self._peak_rss: float = 0.0
        self._peak_sys: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread."""
        self._stop_event.clear()
        self._peak_rss = 0.0
        self._peak_sys = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="MemoryMonitor")
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 10)
            self._thread = None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def peak_process_rss_mb(self) -> float:
        """Peak process RSS seen during the run (MB)."""
        return self._peak_rss / (1024.0 * 1024.0)

    @property
    def peak_system_used_mb(self) -> float:
        """Peak system-wide used memory seen during the run (MB)."""
        return self._peak_sys / (1024.0 * 1024.0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            rss, sys_used = self._sampler()
            if rss > self._peak_rss:
                self._peak_rss = rss
            if sys_used > self._peak_sys:
                self._peak_sys = sys_used
            self._stop_event.wait(timeout=self._interval_s)

    @staticmethod
    def _default_sampler() -> tuple[float, float]:
        proc = psutil.Process()
        rss = float(proc.memory_info().rss)
        sys_used = float(psutil.virtual_memory().used)
        return rss, sys_used
