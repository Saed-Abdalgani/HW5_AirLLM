"""Summary and visualisation helpers — public SDK surface.

Provides:
- ``summarize(results_dir)`` — read all run JSON files into a tidy DataFrame.
- ``plot_comparison(df, out_path)`` — produce bar charts to ``assets/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_METRICS = [
    ("total_runtime_s", "Total Runtime (s)", "comparison_runtime.png"),
    ("peak_process_rss_mb", "Peak Process RSS (MB)", "comparison_peak_rss.png"),
    ("tokens_per_s", "Tokens per Second", "comparison_tokens_per_s.png"),
]


def summarize(results_dir: Path) -> pd.DataFrame:
    """Read all ``run_*.json`` files in *results_dir* into a tidy DataFrame.

    Parameters
    ----------
    results_dir:
        Directory containing ``run_*.json`` files.

    Returns
    -------
    pd.DataFrame
        One row per run; columns are ``RunResult`` fields.
    """
    records: list[dict] = []
    for p in sorted(results_dir.glob("run_*.json")):
        with p.open(encoding="utf-8") as fh:
            records.append(json.load(fh))
    if not records:
        logger.warning("summarize: no run_*.json files found in %s", results_dir)
        return pd.DataFrame()
    df = pd.json_normalize(records)
    logger.info("summarize: loaded %d records", len(df))
    return df


def plot_comparison(df: pd.DataFrame, out_path: Path) -> list[Path]:
    """Generate bar charts for runtime, peak RSS, and tokens/s.

    Failed / N/A backends are shown with hatching and a zero bar.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`summarize`.
    out_path:
        Directory where chart PNGs are written.

    Returns
    -------
    list[Path]
        Paths to the generated PNG files.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for col, ylabel, filename in _METRICS:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        backends = df["backend"].tolist()
        values = df[col].fillna(0).tolist()
        statuses = df["status"].tolist() if "status" in df.columns else ["success"] * len(backends)

        bars = ax.bar(backends, values)
        for bar, status in zip(bars, statuses):
            if status != "success":
                bar.set_hatch("//")
                bar.set_alpha(0.5)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Backend")
        ax.set_title(f"Comparison — {ylabel}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        dest = out_path / filename
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        written.append(dest)
        logger.info("plot_comparison: saved %s", dest)

    return written
