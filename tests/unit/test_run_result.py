"""Unit tests for services.metrics.run_result.RunResult factories."""

from __future__ import annotations

from airllm_bench.constants import STATUS_FAILED, STATUS_SUCCESS
from airllm_bench.services.metrics.run_result import RunResult

HOST = {"cpu": "test", "has_cuda": False}


class TestRunResultFactories:
    def test_success_computes_tokens_per_s(self):
        result = RunResult.success(
            backend="airllm",
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt="Hello",
            max_new_tokens=8,
            load_time_s=1.0,
            generate_time_s=2.0,
            peak_process_rss_mb=500.0,
            peak_system_used_mb=4000.0,
            output="Generated output text",
            host=HOST,
        )
        assert result.status == STATUS_SUCCESS
        assert result.tokens_per_s == 4.0
        assert result.total_runtime_s == 3.0
        assert result.output_preview == "Generated output text"

    def test_failed_sets_reason(self):
        result = RunResult.failed(
            backend="transformers-cpu",
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt="Hello",
            max_new_tokens=8,
            reason="OOM",
            host=HOST,
            load_time_s=10.0,
            peak_process_rss_mb=3400.0,
        )
        assert result.status == STATUS_FAILED
        assert result.failure_reason == "OOM"
        assert result.generate_time_s is None

    def test_to_dict_roundtrip_keys(self):
        result = RunResult.success(
            backend="ollama",
            model_id="qwen2:0.5b",
            prompt="Hi",
            max_new_tokens=4,
            load_time_s=0.5,
            generate_time_s=1.0,
            peak_process_rss_mb=100.0,
            peak_system_used_mb=2000.0,
            output="ok",
            host=HOST,
        )
        data = result.to_dict()
        assert data["backend"] == "ollama"
        assert "timestamp" in data
        assert "airllm_bench_version" in data
