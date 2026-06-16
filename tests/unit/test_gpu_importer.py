"""Unit tests for GpuResultImporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from airllm_bench.services.backends.gpu_importer import GpuResultImporter


def _write_sample(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "run_gpu_sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


VALID_DATA = {
    "backend": "gpu",
    "status": "success",
    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "load_time_s": 1.2,
    "generate_time_s": 3.5,
    "tokens_per_s": 4.6,
    "peak_process_rss_mb": 6144.0,
}


class TestGpuResultImporter:
    def test_load_returns_dict(self, tmp_path):
        p = _write_sample(tmp_path, VALID_DATA)
        importer = GpuResultImporter(p)
        data = importer.load()
        assert data["backend"] == "gpu"
        assert data["status"] == "success"

    def test_load_raises_file_not_found(self, tmp_path):
        importer = GpuResultImporter(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            importer.load()

    def test_load_raises_on_missing_keys(self, tmp_path):
        bad_data = {"backend": "gpu"}  # missing status, model_id, timestamp
        p = _write_sample(tmp_path, bad_data)
        importer = GpuResultImporter(p)
        with pytest.raises(ValueError, match="missing required keys"):
            importer.load()

    def test_make_na_result_has_required_keys(self):
        result = GpuResultImporter.make_na_result("Qwen/Qwen2.5-3B-Instruct")
        assert result["status"] == "n/a"
        assert result["backend"] == "gpu"
        assert result["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
        assert result["tokens_per_s"] is None
