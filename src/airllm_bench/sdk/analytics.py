"""Summary and visualisation helpers — public SDK surface.

Provides:
- ``summarize(results_dir)`` — read run JSON files into a tidy DataFrame + CSV.
- ``plot_comparison(df, out_path)`` — bar charts to ``assets/``.
- ``write_comparison_table(df, results_dir)`` — Markdown table in ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from airllm_bench.sdk.analytics_data import build_comparison_df, write_comparison_csv
from airllm_bench.sdk.analytics_plot import plot_comparison
from airllm_bench.sdk.analytics_table import render_comparison_markdown, write_comparison_table

__all__ = [
    "summarize",
    "plot_comparison",
    "render_comparison_markdown",
    "write_comparison_table",
]


def summarize(results_dir: Path) -> pd.DataFrame:
    """Read all ``run_*.json`` files, merge to one row per backend, write CSV."""
    results_dir = Path(results_dir)
    df = build_comparison_df(results_dir)
    if not df.empty:
        write_comparison_csv(df, results_dir)
    return df
