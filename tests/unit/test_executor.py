"""Unit tests for sdk._executor.execute_run.

Tests cover: success path, MemoryError → OOM reason, RuntimeError with 'memory'
→ OOM reason, TimeoutError → timeout reason, unexpected Exception → unknown reason.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from airllm_bench.constants import REASON_OOM, REASON_TIMEOUT, STATUS_FAILED, STATUS_SUCCESS
from airllm_bench.sdk._executor import execute_run
from airllm_bench.services.metrics.memory_monitor import MemoryMonitor
from airllm_bench.services.metrics.timer import Timer
from airllm_bench.shared.config import Settings

FAKE_HOST = {"cpu": "test-cpu", "total_ram_mb": 8000.0, "has_cuda": False}


def _make_deps():
    settings = Settings()
    monitor = MagicMock(spec=MemoryMonitor)
    monitor.peak_process_rss_mb = 500.0
    monitor.peak_system_used_mb = 4000.0
    timer = MagicMock(spec=Timer)
    timer.load_time_s = 1.0
    timer.generate_time_s = 2.0
    # Make phase() work as a context manager
    timer.phase.return_value.__enter__ = MagicMock(return_value=None)
    timer.phase.return_value.__exit__ = MagicMock(return_value=False)
    return settings, monitor, timer


class TestExecuteRunSuccess:
    def test_success_path_returns_success_result(self):
        settings, monitor, timer = _make_deps()

        backend = MagicMock()
        backend.name = "ollama"
        backend.generate.return_value = "Virtual memory explanation text"

        result = execute_run(backend, settings, monitor, timer, FAKE_HOST)

        assert result.status == STATUS_SUCCESS
        assert result.backend == "ollama"
        assert result.output_preview is not None
        backend.load.assert_called_once()
        backend.generate.assert_called_once()
        backend.teardown.assert_called_once()

    def test_success_path_calls_monitor_start_and_stop(self):
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "ollama"
        backend.generate.return_value = "ok"

        execute_run(backend, settings, monitor, timer, FAKE_HOST)

        monitor.start.assert_called_once()
        monitor.stop.assert_called_once()


class TestExecuteRunFailures:
    def test_memory_error_maps_to_oom(self):
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "transformers-cpu"
        backend.load.side_effect = MemoryError("out of memory")

        result = execute_run(backend, settings, monitor, timer, FAKE_HOST)

        assert result.status == STATUS_FAILED
        assert result.failure_reason == REASON_OOM
        backend.teardown.assert_called_once()

    def test_runtime_error_with_memory_maps_to_oom(self):
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "transformers-cpu"
        backend.load.side_effect = RuntimeError("not enough memory: allocate 6GB")

        result = execute_run(backend, settings, monitor, timer, FAKE_HOST)

        assert result.status == STATUS_FAILED
        assert result.failure_reason == REASON_OOM

    def test_timeout_error_maps_to_timeout_reason(self):
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "airllm"
        backend.generate.side_effect = TimeoutError("timed out")
        # load succeeds, generate times out
        backend.load.return_value = None

        result = execute_run(backend, settings, monitor, timer, FAKE_HOST)

        assert result.status == STATUS_FAILED
        assert result.failure_reason == REASON_TIMEOUT

    def test_unexpected_exception_captured(self):
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "airllm"
        backend.load.side_effect = ValueError("something unexpected")

        result = execute_run(backend, settings, monitor, timer, FAKE_HOST)

        assert result.status == STATUS_FAILED
        assert "ValueError" in result.failure_reason

    def test_teardown_always_called(self):
        """teardown() must be called even on exception paths."""
        settings, monitor, timer = _make_deps()
        backend = MagicMock()
        backend.name = "ollama"
        backend.load.side_effect = MemoryError("oom")

        execute_run(backend, settings, monitor, timer, FAKE_HOST)

        backend.teardown.assert_called_once()
