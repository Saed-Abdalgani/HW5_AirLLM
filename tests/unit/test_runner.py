"""Unit tests for sdk.runner.BenchmarkRunner.

All heavy dependencies (backends, executor, host_spec) are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from airllm_bench.constants import BackendName, STATUS_SUCCESS
from airllm_bench.sdk.runner import BenchmarkRunner
from airllm_bench.services.metrics.run_result import RunResult
from airllm_bench.shared.config import Settings

FAKE_HOST = {"cpu": "test-cpu", "total_ram_mb": 8000.0, "has_cuda": False}


def _fake_result(backend: str = "ollama") -> RunResult:
    return RunResult.success(
        backend=backend,
        model_id="test/model",
        prompt="Hello",
        max_new_tokens=8,
        load_time_s=1.0,
        generate_time_s=2.0,
        peak_process_rss_mb=200.0,
        peak_system_used_mb=4000.0,
        output="Generated text",
        host=FAKE_HOST,
    )


class TestBenchmarkRunnerRun:
    def test_run_returns_result_and_writes_json(self, tmp_path):
        settings = Settings(results_dir=str(tmp_path))

        with (
            patch("airllm_bench.sdk.runner.capture_host_spec", return_value=FAKE_HOST),
            patch("airllm_bench.sdk.runner.write_host_spec"),
            patch("airllm_bench.sdk.runner.execute_run", return_value=_fake_result("ollama")),
        ):
            runner = BenchmarkRunner(settings)
            result = runner.run(BackendName.OLLAMA)

        assert result.status == STATUS_SUCCESS
        assert result.backend == "ollama"
        # JSON result file should exist in results dir
        json_files = list(tmp_path.glob("run_ollama_*.json"))
        assert len(json_files) == 1

    def test_run_all_collects_results(self, tmp_path):
        settings = Settings(results_dir=str(tmp_path))

        fake_ollama = _fake_result("ollama")
        fake_airllm = _fake_result("airllm")

        results_seq = [fake_ollama, fake_airllm]

        with (
            patch("airllm_bench.sdk.runner.capture_host_spec", return_value=FAKE_HOST),
            patch("airllm_bench.sdk.runner.write_host_spec"),
            patch("airllm_bench.sdk.runner.execute_run", side_effect=results_seq),
        ):
            runner = BenchmarkRunner(settings)
            results = runner.run_all([BackendName.OLLAMA, BackendName.AIRLLM])

        assert len(results) == 2
        assert results[0].backend == "ollama"
        assert results[1].backend == "airllm"

    def test_run_all_continues_after_failure(self, tmp_path):
        """A failed backend should not prevent subsequent backends from running."""
        from airllm_bench.constants import STATUS_FAILED

        settings = Settings(results_dir=str(tmp_path))

        failed_result = RunResult.failed(
            backend="transformers-cpu",
            model_id="test/model",
            prompt="Hello",
            max_new_tokens=8,
            reason="OOM",
            host=FAKE_HOST,
        )
        success_result = _fake_result("airllm")
        results_seq = [failed_result, success_result]

        with (
            patch("airllm_bench.sdk.runner.capture_host_spec", return_value=FAKE_HOST),
            patch("airllm_bench.sdk.runner.write_host_spec"),
            patch("airllm_bench.sdk.runner.execute_run", side_effect=results_seq),
        ):
            runner = BenchmarkRunner(settings)
            results = runner.run_all([BackendName.TRANSFORMERS_CPU, BackendName.AIRLLM])

        assert results[0].status == STATUS_FAILED
        assert results[1].status == STATUS_SUCCESS


class TestBenchmarkRunnerBuildBackend:
    def test_build_backend_unknown_raises(self, tmp_path):
        settings = Settings(results_dir=str(tmp_path))
        runner = BenchmarkRunner(settings)
        with pytest.raises(ValueError, match="Unknown backend"):
            runner._build_backend("not-a-real-backend")  # noqa: SLF001
