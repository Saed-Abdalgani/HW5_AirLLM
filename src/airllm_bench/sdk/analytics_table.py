"""Markdown comparison table rendering for benchmark results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from airllm_bench.sdk.analytics_data import COMPARISON_TABLE_FILENAME


def _fmt_num(value: object, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _fmt_status(status: str, reason: str) -> str:
    if status == "success":
        return "success"
    if status == "failed":
        return f"failed ({reason})" if reason else "failed"
    if status == "n/a":
        return "n/a"
    return status


def render_comparison_markdown(df: pd.DataFrame) -> str:
    """Render a Markdown table covering GPU, CPU, AirLLM, and Ollama backends."""
    header = (
        "| Backend | Model | Status | Response Time (s) | "
        "Peak Memory (MB) | Total Runtime (s) | Tokens/s |\n"
        "|---------|-------|--------|-------------------|"
        "------------------|-------------------|----------|"
    )
    if df.empty:
        return header + "\n| — | — | — | — | — | — | — |"

    lines = [header]
    for _, row in df.iterrows():
        model = str(row.get("model_id", "")).split("/")[-1]
        lines.append(
            "| {backend} | {model} | {status} | {resp} | {peak} | {total} | {tps} |".format(
                backend=row.get("display_name", row.get("backend", "")),
                model=model,
                status=_fmt_status(str(row.get("status", "")), str(row.get("failure_reason", ""))),
                resp=_fmt_num(row.get("generate_time_s")),
                peak=_fmt_num(row.get("peak_process_rss_mb"), 1),
                total=_fmt_num(row.get("total_runtime_s")),
                tps=_fmt_num(row.get("tokens_per_s"), 3),
            )
        )
    return "\n".join(lines)


def write_comparison_table(df: pd.DataFrame, results_dir: Path) -> Path:
    """Write the Markdown comparison table to the results directory."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / COMPARISON_TABLE_FILENAME
    body = (
        "# Benchmark Comparison\n\n"
        "Cross-backend comparison of response time, peak memory, and total runtime.\n\n"
        f"{render_comparison_markdown(df)}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path
