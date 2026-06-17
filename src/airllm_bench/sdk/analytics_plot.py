"""Bar-chart generation for benchmark comparison metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_CHART_SPECS: list[tuple[str, str, str]] = [
    ("total_runtime_s", "Total Runtime (s)", "comparison_runtime.png"),
    ("peak_process_rss_mb", "Peak Process RSS (MB)", "comparison_peak_rss.png"),
    ("tokens_per_s", "Tokens per Second", "comparison_tokens_per_s.png"),
]

_ANNOTATION: dict[str, str] = {
    "failed": "failed",
    "n/a": "n/a",
}


def _annotate_bars(ax, bars, statuses: list[str], reasons: list[str]) -> None:
    for bar, status, reason in zip(bars, statuses, reasons, strict=True):
        if status == "success":
            continue
        bar.set_hatch("//")
        bar.set_alpha(0.5)
        label = reason if status == "failed" and reason else _ANNOTATION.get(status, status)
        height = bar.get_height()
        y = max(height, 0.05)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def plot_comparison(df: pd.DataFrame, out_path: Path) -> list[Path]:
    """Generate runtime, peak-RSS, and tokens/s bar charts."""
    out_path.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.warning("plot_comparison: empty DataFrame, skipping charts")
        return []

    labels = df["display_name"].fillna(df["backend"]).tolist()
    statuses = df["status"].tolist() if "status" in df.columns else ["success"] * len(df)
    reasons = df["failure_reason"].fillna("").tolist() if "failure_reason" in df.columns else [""] * len(df)
    written: list[Path] = []

    for col, ylabel, filename in _CHART_SPECS:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        values = [0.0 if pd.isna(v) else float(v) for v in df[col].tolist()]
        bars = ax.bar(labels, values)
        _annotate_bars(ax, bars, statuses, reasons)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Backend")
        ax.set_title(f"Comparison — {ylabel}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        dest = out_path / filename
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        written.append(dest)
        logger.info("plot_comparison: saved %s", dest)

    return written
