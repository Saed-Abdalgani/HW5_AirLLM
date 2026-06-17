"""BenchmarkRunner — the public SDK entry point for orchestrating runs.

CLI and notebooks call *only* this class (and the module-level helpers
``summarize`` and ``plot_comparison``).  No service is imported directly by
callers outside the ``sdk`` package.

The run-execution logic (timer + monitor wiring + error mapping) lives in
:mod:`airllm_bench.sdk._executor` to keep this file under 150 lines.
"""

from __future__ import annotations

import logging
from pathlib import Path

from airllm_bench.constants import BackendName
from airllm_bench.sdk._executor import execute_run
from airllm_bench.services.backends.airllm_backend import AirllmBackend
from airllm_bench.services.backends.ollama_backend import OllamaBackend
from airllm_bench.services.backends.transformers_cpu_backend import TransformersCpuBackend
from airllm_bench.services.host_spec import capture_host_spec, write_host_spec
from airllm_bench.services.metrics.memory_monitor import MemoryMonitor
from airllm_bench.services.metrics.recorder import MetricsRecorder, RunResult
from airllm_bench.services.metrics.timer import Timer
from airllm_bench.shared.config import Settings
from airllm_bench.shared.gatekeeper import Gatekeeper

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrate benchmark runs for one or more backends.

    Parameters
    ----------
    settings:
        Application settings (model IDs, prompt, token counts, etc.).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._gatekeeper = Gatekeeper(settings)
        self._recorder = MetricsRecorder(Path(settings.results_dir))

    # ------------------------------------------------------------------
    # Public SDK API
    # ------------------------------------------------------------------

    def run(self, backend: BackendName) -> RunResult:
        """Run a single backend and return its ``RunResult``.

        Parameters
        ----------
        backend:
            Which backend to exercise.

        Returns
        -------
        RunResult
            A success or failed result — never raises.
        """
        host = capture_host_spec()
        write_host_spec(Path(self._settings.results_dir))

        backend_impl = self._build_backend(backend)
        monitor = MemoryMonitor(interval_s=self._settings.sample_interval_s)
        timer = Timer()

        result = execute_run(backend_impl, self._settings, monitor, timer, host)
        self._recorder.write(result)
        return result

    def run_all(self, backends: list[BackendName]) -> list[RunResult]:
        """Run every backend in *backends* sequentially, collecting results.

        A failure in one backend does not stop subsequent backends.
        """
        results: list[RunResult] = []
        for b in backends:
            logger.info("BenchmarkRunner: starting %s …", b.value)
            results.append(self.run(b))
        return results

    def import_gpu_result(self, colab_json_path: Path | None = None) -> RunResult:
        """Import a Colab GPU JSON result or record an N/A estimate if not found."""
        import dataclasses

        from airllm_bench.services.backends.gpu_importer import GpuResultImporter
        if colab_json_path and colab_json_path.exists():
            importer = GpuResultImporter(colab_json_path)
            data = importer.load()
        else:
            data = GpuResultImporter.make_na_result(self._settings.model_id)

        valid_keys = {f.name for f in dataclasses.fields(RunResult)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        result = RunResult(**filtered_data)

        self._recorder.write(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_backend(self, backend: BackendName):  # noqa: ANN202
        """Instantiate the correct backend implementation."""
        if backend == BackendName.OLLAMA:
            return OllamaBackend(self._settings, self._gatekeeper)
        if backend == BackendName.TRANSFORMERS_CPU:
            return TransformersCpuBackend(self._settings)
        if backend == BackendName.AIRLLM:
            return AirllmBackend(self._settings)
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
