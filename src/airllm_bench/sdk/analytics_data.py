"""Data loading and tabular helpers for benchmark comparison analytics."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pandas as pd

from airllm_bench.constants import COMPARISON_CSV_FILENAME

logger = logging.getLogger(__name__)

COMPARISON_COLUMNS: list[str] = [
    "backend",
    "display_name",
    "model_id",
    "status",
    "failure_reason",
    "load_time_s",
    "generate_time_s",
    "total_runtime_s",
    "tokens_per_s",
    "peak_process_rss_mb",
    "peak_system_used_mb",
    "timestamp",
]

BACKEND_DISPLAY: dict[str, str] = {
    "gpu": "GPU",
    "transformers-cpu": "CPU",
    "airllm": "AirLLM",
    "ollama": "Ollama",
}

BACKEND_ORDER: list[str] = ["gpu", "transformers-cpu", "airllm", "ollama"]

COMPARISON_TABLE_FILENAME = "comparison_table.md"


def _load_run_records(results_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(results_dir.glob("run_*.json")):
        with path.open(encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def _row_from_record(record: dict) -> dict:
    backend = record.get("backend", "")
    row = {col: record.get(col) for col in COMPARISON_COLUMNS if col != "display_name"}
    row["display_name"] = BACKEND_DISPLAY.get(backend, backend)
    row["failure_reason"] = record.get("failure_reason") or ""
    return row


def _dedupe_latest(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        backend = row["backend"]
        prev = latest.get(backend)
        if prev is None or row.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[backend] = row
    order = {name: idx for idx, name in enumerate(BACKEND_ORDER)}
    return sorted(latest.values(), key=lambda r: order.get(r["backend"], 99))


def build_comparison_df(results_dir: Path) -> pd.DataFrame:
    """Load run JSON files, dedupe to one row per backend, return tidy DataFrame."""
    records = _load_run_records(results_dir)
    if not records:
        logger.warning("build_comparison_df: no run_*.json in %s", results_dir)
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    rows = _dedupe_latest([_row_from_record(r) for r in records])
    df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    logger.info("build_comparison_df: %d backends from %d files", len(df), len(records))
    return df


def write_comparison_csv(df: pd.DataFrame, results_dir: Path) -> Path:
    """Write comparison CSV with a stable column set."""
    csv_path = results_dir / COMPARISON_CSV_FILENAME
    results_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in COMPARISON_COLUMNS if c != "display_name"]
    df[export_cols].to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info("write_comparison_csv: saved %s", csv_path)
    return csv_path
