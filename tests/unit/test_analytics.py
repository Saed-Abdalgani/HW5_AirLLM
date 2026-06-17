"""Unit tests for sdk analytics (summarize, plot, comparison table)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from airllm_bench.constants import COMPARISON_CSV_FILENAME, STATUS_FAILED, STATUS_NA, STATUS_SUCCESS
from airllm_bench.sdk.analytics import (
    plot_comparison,
    render_comparison_markdown,
    summarize,
    write_comparison_table,
)
from airllm_bench.sdk.analytics_data import COMPARISON_COLUMNS, COMPARISON_TABLE_FILENAME


def _write_run(tmp_path: Path, name: str, payload: dict) -> None:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_record(backend: str, status: str, **overrides) -> dict:
    record = {
        "backend": backend,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "status": status,
        "failure_reason": None,
        "load_time_s": 1.0,
        "generate_time_s": 2.0,
        "total_runtime_s": 3.0,
        "tokens_per_s": 4.0,
        "peak_process_rss_mb": 500.0,
        "peak_system_used_mb": 6000.0,
        "timestamp": "2026-06-15T10:00:00+00:00",
    }
    record.update(overrides)
    return record


class TestSummarize:
    def test_mixed_statuses_stable_columns(self, tmp_path: Path):
        _write_run(tmp_path, "run_gpu_a.json", _base_record("gpu", STATUS_SUCCESS))
        _write_run(
            tmp_path,
            "run_transformers_cpu_a.json",
            _base_record("transformers-cpu", STATUS_FAILED, failure_reason="OOM", generate_time_s=None),
        )
        _write_run(tmp_path, "run_airllm_a.json", _base_record("airllm", STATUS_SUCCESS))
        _write_run(
            tmp_path,
            "run_ollama_a.json",
            _base_record("ollama", STATUS_NA, model_id="qwen2:0.5b", generate_time_s=None),
        )

        df = summarize(tmp_path)

        assert list(df.columns) == COMPARISON_COLUMNS
        assert len(df) == 4
        assert set(df["backend"]) == {"gpu", "transformers-cpu", "airllm", "ollama"}

        csv_path = tmp_path / COMPARISON_CSV_FILENAME
        assert csv_path.exists()
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == 4
        assert "status" in csv_df.columns

    def test_dedupes_to_latest_per_backend(self, tmp_path: Path):
        _write_run(
            tmp_path,
            "run_airllm_old.json",
            _base_record("airllm", STATUS_SUCCESS, total_runtime_s=100.0, timestamp="2026-06-01T00:00:00+00:00"),
        )
        _write_run(
            tmp_path,
            "run_airllm_new.json",
            _base_record("airllm", STATUS_SUCCESS, total_runtime_s=200.0, timestamp="2026-06-15T00:00:00+00:00"),
        )

        df = summarize(tmp_path)

        assert len(df) == 1
        assert df.iloc[0]["total_runtime_s"] == 200.0

    def test_empty_dir_returns_empty_frame(self, tmp_path: Path):
        df = summarize(tmp_path)
        assert df.empty


class TestPlotComparison:
    def test_png_files_created(self, tmp_path: Path):
        df = pd.DataFrame([
            _base_record("gpu", STATUS_SUCCESS, display_name="GPU"),
            _base_record("transformers-cpu", STATUS_FAILED, failure_reason="OOM", display_name="CPU"),
        ])
        out = tmp_path / "assets"
        paths = plot_comparison(df, out)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()
            assert path.suffix == ".png"
            assert path.stat().st_size > 0

    def test_empty_dataframe_returns_no_paths(self, tmp_path: Path):
        paths = plot_comparison(pd.DataFrame(), tmp_path / "assets")
        assert paths == []


class TestComparisonTable:
    def test_markdown_includes_all_backends(self, tmp_path: Path):
        df = pd.DataFrame([
            {**_base_record("gpu", STATUS_SUCCESS), "display_name": "GPU"},
            {**_base_record("transformers-cpu", STATUS_FAILED, failure_reason="OOM"), "display_name": "CPU"},
            {**_base_record("airllm", STATUS_SUCCESS), "display_name": "AirLLM"},
            {**_base_record("ollama", STATUS_SUCCESS, model_id="qwen2:0.5b"), "display_name": "Ollama"},
        ])

        md = render_comparison_markdown(df)
        assert "GPU" in md
        assert "CPU" in md
        assert "AirLLM" in md
        assert "Ollama" in md
        assert "failed (OOM)" in md

        path = write_comparison_table(df, tmp_path)
        assert path.name == COMPARISON_TABLE_FILENAME
        assert path.read_text(encoding="utf-8").startswith("# Benchmark Comparison")

    def test_na_status_rendered(self):
        df = pd.DataFrame([
            {**_base_record("gpu", STATUS_NA), "display_name": "GPU"},
        ])
        md = render_comparison_markdown(df)
        assert "n/a" in md

    def test_empty_dataframe_header_only(self):
        md = render_comparison_markdown(pd.DataFrame())
        assert "Backend | Model | Status" in md
        assert "—" in md
