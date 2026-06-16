"""Unit tests for metrics.recorder (RunResult + MetricsRecorder)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from airllm_bench.constants import STATUS_FAILED, STATUS_SUCCESS
from airllm_bench.services.metrics.recorder import MetricsRecorder, RunResult


FAKE_HOST = {"cpu": "Intel i3", "total_ram_mb": 8192.0, "has_cuda": False}


class TestRunResultFactories:
    def test_success_factory_sets_status(self):
        r = RunResult.success(
            backend="ollama",
            model_id="test/model",
            prompt="Hello",
            max_new_tokens=16,
            load_time_s=1.5,
            generate_time_s=2.0,
            peak_process_rss_mb=800.0,
            peak_system_used_mb=4000.0,
            output="World",
            host=FAKE_HOST,
        )
        assert r.status == STATUS_SUCCESS
        assert r.tokens_per_s == pytest.approx(16 / 2.0)
        assert r.total_runtime_s == pytest.approx(1.5 + 2.0)
        assert r.failure_reason is None

    def test_success_output_preview_truncated(self):
        long_output = "x" * 300
        r = RunResult.success(
            backend="ollama",
            model_id="test/model",
            prompt="Hi",
            max_new_tokens=8,
            load_time_s=0.1,
            generate_time_s=0.5,
            peak_process_rss_mb=100.0,
            peak_system_used_mb=500.0,
            output=long_output,
            host=FAKE_HOST,
        )
        assert len(r.output_preview) <= 200

    def test_failed_factory_sets_status(self):
        r = RunResult.failed(
            backend="transformers-cpu",
            model_id="test/model",
            prompt="Hi",
            max_new_tokens=16,
            reason="OOM",
            host=FAKE_HOST,
        )
        assert r.status == STATUS_FAILED
        assert r.failure_reason == "OOM"
        assert r.tokens_per_s is None
        assert r.output_preview is None

    def test_to_dict_contains_required_keys(self):
        r = RunResult.success(
            backend="airllm",
            model_id="test/model",
            prompt="Test",
            max_new_tokens=4,
            load_time_s=0.1,
            generate_time_s=0.1,
            peak_process_rss_mb=50.0,
            peak_system_used_mb=200.0,
            output="Hi",
            host=FAKE_HOST,
        )
        d = r.to_dict()
        required = {"backend", "model_id", "status", "timestamp", "host", "airllm_bench_version"}
        assert required.issubset(d.keys())


class TestMetricsRecorder:
    def test_write_creates_json_file(self, tmp_path):
        recorder = MetricsRecorder(tmp_path)
        r = RunResult.success(
            backend="ollama",
            model_id="m",
            prompt="p",
            max_new_tokens=4,
            load_time_s=0.1,
            generate_time_s=0.1,
            peak_process_rss_mb=10.0,
            peak_system_used_mb=100.0,
            output="ok",
            host=FAKE_HOST,
        )
        json_path = recorder.write(r)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["backend"] == "ollama"
        assert data["status"] == STATUS_SUCCESS

    def test_write_creates_csv(self, tmp_path):
        recorder = MetricsRecorder(tmp_path)
        r = RunResult.failed(
            backend="transformers-cpu",
            model_id="m",
            prompt="p",
            max_new_tokens=4,
            reason="OOM",
            host=FAKE_HOST,
        )
        recorder.write(r)
        csv_path = tmp_path / "comparison.csv"
        assert csv_path.exists()
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["backend"] == "transformers-cpu"

    def test_write_multiple_runs_appends_rows(self, tmp_path):
        recorder = MetricsRecorder(tmp_path)
        for i in range(3):
            r = RunResult.success(
                backend=f"backend_{i}",
                model_id="m",
                prompt="p",
                max_new_tokens=4,
                load_time_s=float(i),
                generate_time_s=0.1,
                peak_process_rss_mb=10.0,
                peak_system_used_mb=100.0,
                output="ok",
                host=FAKE_HOST,
            )
            recorder.write(r)
        csv_path = tmp_path / "comparison.csv"
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3

    def test_results_dir_created_if_absent(self, tmp_path):
        new_dir = tmp_path / "new_results"
        assert not new_dir.exists()
        MetricsRecorder(new_dir)
        assert new_dir.exists()
