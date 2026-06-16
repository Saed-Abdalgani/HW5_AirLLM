"""GPU result importer (B4) — loads Colab-produced RunResult JSON.

Because the host has no CUDA GPU (Intel UHD, no CUDA — see ADR-2), the GPU
reference run is executed on Google Colab (free T4) using the same SDK path.
The resulting JSON is downloaded into ``results/`` and loaded here so it joins
the comparison table like any other backend.

If no Colab JSON is available, a placeholder N/A RunResult is produced with
a theoretical estimate and documented rationale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from airllm_bench.constants import BackendName, STATUS_NA

logger = logging.getLogger(__name__)


class GpuResultImporter:
    """Import a Colab-produced RunResult JSON file into the comparison set.

    Parameters
    ----------
    json_path:
        Path to the ``run_gpu_*.json`` file exported from the Colab notebook.
    """

    def __init__(self, json_path: Path) -> None:
        self._path = json_path

    def load(self) -> dict:
        """Read and return the RunResult dict from the JSON file.

        Returns
        -------
        dict
            The full RunResult dict (same schema as locally-produced results).

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        ValueError
            If the JSON does not contain the expected schema keys.
        """
        if not self._path.exists():
            msg = f"GPU result file not found: {self._path}"
            raise FileNotFoundError(msg)

        with self._path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        required = {"backend", "status", "model_id", "timestamp"}
        missing = required - set(data.keys())
        if missing:
            msg = f"GPU result JSON missing required keys: {missing}"
            raise ValueError(msg)

        logger.info("GpuResultImporter: loaded %s", self._path.name)
        return data

    @staticmethod
    def make_na_result(model_id: str, reason: str = "No CUDA device on host") -> dict:
        """Produce a placeholder N/A RunResult when Colab is unavailable.

        Parameters
        ----------
        model_id:
            Model that would have been benchmarked.
        reason:
            Explanation of why the GPU run is not available.
        """
        import datetime

        return {
            "backend": BackendName.GPU.value,
            "model_id": model_id,
            "status": STATUS_NA,
            "failure_reason": reason,
            "load_time_s": None,
            "ttft_s": None,
            "generate_time_s": None,
            "total_runtime_s": None,
            "tokens_per_s": None,
            "peak_process_rss_mb": None,
            "peak_system_used_mb": None,
            "output_preview": None,
            "host": {"has_cuda": False, "device": "N/A", "note": reason},
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "airllm_bench_version": "0.1.0",
            "note": "Theoretical: T4 fp16 ≈ 1–2 s load, 5–10 tokens/s, ~6 GB VRAM peak.",
        }
