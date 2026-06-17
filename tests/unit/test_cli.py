"""Unit tests for cli.py using typer CliRunner.

SDK calls are mocked so no real inference runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from airllm_bench.cli import app
from airllm_bench.constants import STATUS_FAILED, STATUS_SUCCESS
from airllm_bench.services.metrics.run_result import RunResult

runner = CliRunner()

FAKE_HOST = {"cpu": "test-cpu", "total_ram_mb": 8000.0, "has_cuda": False}


def _success_result(backend: str = "ollama") -> RunResult:
    return RunResult.success(
        backend=backend,
        model_id="qwen2:0.5b",
        prompt="Explain virtual memory in one sentence.",
        max_new_tokens=16,
        load_time_s=2.85,
        generate_time_s=4.21,
        peak_process_rss_mb=612.4,
        peak_system_used_mb=5847.2,
        output="Virtual memory is a memory management technique.",
        host=FAKE_HOST,
    )


class TestCliRun:
    def test_run_ollama_prints_backend(self):
        mock_result = _success_result("ollama")
        with patch("airllm_bench.cli.BenchmarkRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            result = runner.invoke(app, ["run", "--backend", "ollama"])
        assert result.exit_code == 0
        assert "ollama" in result.output.lower() or "Backend" in result.output

    def test_run_with_model_id_override(self):
        mock_result = _success_result("ollama")
        with patch("airllm_bench.cli.BenchmarkRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            result = runner.invoke(app, ["run", "--backend", "ollama", "--model-id", "qwen2:0.5b"])
        assert result.exit_code == 0


class TestCliRunAll:
    def test_run_all_calls_runner(self):
        results = [_success_result("ollama"), _success_result("airllm")]
        with patch("airllm_bench.cli.BenchmarkRunner") as MockRunner:
            MockRunner.return_value.run_all.return_value = results
            result = runner.invoke(app, ["run-all"])
        assert result.exit_code == 0


class TestCliReport:
    def test_report_no_results_exits_nonzero(self, tmp_path):
        import pandas as pd

        with (
            patch("airllm_bench.cli.get_settings") as mock_settings,
            patch("airllm_bench.cli.summarize", return_value=pd.DataFrame()),
        ):
            mock_settings.return_value.results_dir = str(tmp_path)
            mock_settings.return_value.assets_dir = str(tmp_path)
            result = runner.invoke(app, ["report"])
        assert result.exit_code != 0

    def test_report_with_results_prints_table(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame([{
            "backend": "ollama",
            "status": STATUS_SUCCESS,
            "load_time_s": 2.85,
            "generate_time_s": 4.21,
            "total_runtime_s": 7.06,
            "tokens_per_s": 3.8,
            "peak_process_rss_mb": 612.4,
        }])

        with (
            patch("airllm_bench.cli.get_settings") as mock_settings,
            patch("airllm_bench.cli.summarize", return_value=df),
            patch("airllm_bench.cli.plot_comparison", return_value=[]),
            patch("airllm_bench.cli.write_comparison_table", return_value=tmp_path / "comparison_table.md"),
            patch("airllm_bench.cli.render_comparison_markdown", return_value="| Backend | ... |"),
        ):
            mock_settings.return_value.results_dir = str(tmp_path)
            mock_settings.return_value.assets_dir = str(tmp_path)
            result = runner.invoke(app, ["report"])
        assert result.exit_code == 0
        assert "ollama" in result.output


class TestCliHostSpec:
    def test_host_spec_prints_json(self, tmp_path):
        fake_spec = {"cpu": "Intel i5", "has_cuda": False}
        with (
            patch("airllm_bench.cli.get_settings") as mock_settings,
            patch("airllm_bench.cli.capture_host_spec", return_value=fake_spec),
            patch("airllm_bench.cli.write_host_spec", return_value=tmp_path / "host_spec.json"),
        ):
            mock_settings.return_value.results_dir = str(tmp_path)
            result = runner.invoke(app, ["host-spec"])
        assert result.exit_code == 0
        assert "Intel i5" in result.output
