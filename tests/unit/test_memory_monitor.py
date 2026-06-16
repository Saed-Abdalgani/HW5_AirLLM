"""Unit tests for metrics.MemoryMonitor."""

from __future__ import annotations

import time

from airllm_bench.services.metrics.memory_monitor import MemoryMonitor


class TestMemoryMonitorWithFakeSampler:
    """Use an injected fake sampler to avoid real psutil calls."""

    def _make_monitor(self, samples: list[tuple[float, float]]) -> MemoryMonitor:
        """Return a MemoryMonitor whose sampler cycles through *samples*."""
        it = iter(samples)

        def sampler() -> tuple[float, float]:
            try:
                return next(it)
            except StopIteration:
                return (0.0, 0.0)

        return MemoryMonitor(interval_s=0.01, sampler=sampler)

    def test_peak_equals_max_of_samples(self):
        samples = [
            (100 * 1024 * 1024, 2000 * 1024 * 1024),
            (200 * 1024 * 1024, 3000 * 1024 * 1024),
            (150 * 1024 * 1024, 2500 * 1024 * 1024),
        ]
        m = self._make_monitor(samples)
        m.start()
        time.sleep(0.05)
        m.stop()

        assert m.peak_process_rss_mb >= 200.0 - 1.0
        assert m.peak_system_used_mb >= 2500.0 - 1.0

    def test_starts_at_zero_before_start(self):
        m = MemoryMonitor(interval_s=0.01)
        assert m.peak_process_rss_mb == 0.0
        assert m.peak_system_used_mb == 0.0

    def test_stop_is_idempotent(self):
        m = self._make_monitor([(1024 * 1024, 1024 * 1024)])
        m.start()
        m.stop()
        m.stop()  # second stop should not raise

    def test_thread_stops_cleanly(self):
        m = self._make_monitor([])
        m.start()
        m.stop()
        assert m._thread is None  # noqa: SLF001

    def test_peak_resets_on_restart(self):
        samples_first = [(500 * 1024 * 1024, 1000 * 1024 * 1024)]
        m = self._make_monitor(samples_first)
        m.start()
        time.sleep(0.03)
        m.stop()
        peak_after_first = m.peak_process_rss_mb

        # Now restart with lower samples
        def low_sampler():
            return (10 * 1024 * 1024, 10 * 1024 * 1024)

        m._sampler = low_sampler  # noqa: SLF001
        m.start()
        time.sleep(0.03)
        m.stop()

        # Peak should reflect the second run only
        assert m.peak_process_rss_mb < peak_after_first
