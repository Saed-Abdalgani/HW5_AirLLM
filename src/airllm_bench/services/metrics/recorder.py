"""MetricsRecorder — persists RunResult objects to JSON and CSV.

``RunResult`` has been moved to :mod:`airllm_bench.services.metrics.run_result`
to keep each file under 150 lines.  This module re-exports ``RunResult`` for
backwards-compatibility so existing imports still work.

Results are stored as:
- ``results/run_<backend>_<timestamp>.json`` — one record per run.
- ``results/comparison.csv`` — upserted table with one row per run (keyed by
  backend + timestamp).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from airllm_bench.constants import COMPARISON_CSV_FILENAME, RESULT_FILENAME_TEMPLATE
from airllm_bench.services.metrics.run_result import RunResult

# Re-export so callers that do ``from airllm_bench.services.metrics.recorder import RunResult``
# continue to work without modification.
__all__ = ["MetricsRecorder", "RunResult"]

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """Persist RunResult objects to JSON and CSV in the results directory.

    Parameters
    ----------
    results_dir:
        Directory where output files are written (created if absent).
    """

    def __init__(self, results_dir: Path) -> None:
        self._dir = results_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: RunResult) -> Path:
        """Write *result* to a timestamped JSON file and upsert comparison CSV.

        Parameters
        ----------
        result:
            The completed (success or failed) run result.

        Returns
        -------
        Path
            Path to the newly-written JSON file.
        """
        ts_compact = result.timestamp.replace(":", "").replace("-", "").replace("+", "")[:15]
        filename = RESULT_FILENAME_TEMPLATE.format(
            backend=result.backend.replace("-", "_"),
            timestamp=ts_compact,
        )
        json_path = self._dir / filename
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info("MetricsRecorder: wrote %s", json_path)

        self._upsert_csv(result)
        return json_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_csv(self, result: RunResult) -> None:
        """Append or replace one row in the comparison CSV."""
        csv_path = self._dir / COMPARISON_CSV_FILENAME
        row = {
            "backend": result.backend,
            "model_id": result.model_id,
            "status": result.status,
            "failure_reason": result.failure_reason or "",
            "load_time_s": result.load_time_s,
            "generate_time_s": result.generate_time_s,
            "total_runtime_s": result.total_runtime_s,
            "tokens_per_s": result.tokens_per_s,
            "peak_process_rss_mb": result.peak_process_rss_mb,
            "peak_system_used_mb": result.peak_system_used_mb,
            "timestamp": result.timestamp,
        }
        fieldnames = list(row.keys())
        existing = self._read_csv(csv_path)

        # Replace existing row for same backend+timestamp, or append.
        key = (result.backend, result.timestamp)
        existing = [r for r in existing if (r.get("backend"), r.get("timestamp")) != key]
        existing.append(row)

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        logger.debug("MetricsRecorder: updated %s", csv_path)

    @staticmethod
    def _read_csv(csv_path: Path) -> list[dict]:
        """Read an existing CSV file, returning an empty list if absent."""
        if not csv_path.exists():
            return []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
