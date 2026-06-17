"""RunResult — structured outcome of a single benchmark run.

Extracted from ``recorder.py`` to keep each file under 150 lines.
:class:`MetricsRecorder` lives in ``recorder.py`` and imports this class.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from airllm_bench.constants import STATUS_FAILED, STATUS_SUCCESS
from airllm_bench.shared.version import __version__


@dataclasses.dataclass
class RunResult:
    """Structured outcome of one benchmark run."""

    backend: str
    model_id: str
    prompt_chars: int
    max_new_tokens: int
    status: str  # STATUS_SUCCESS | STATUS_FAILED | STATUS_NA
    failure_reason: str | None
    load_time_s: float | None
    ttft_s: float | None
    generate_time_s: float | None
    total_runtime_s: float | None
    tokens_per_s: float | None
    peak_process_rss_mb: float | None
    peak_system_used_mb: float | None
    output_preview: str | None
    host: dict
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat()
    )
    airllm_bench_version: str = dataclasses.field(default=__version__)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        *,
        backend: str,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        load_time_s: float,
        generate_time_s: float,
        peak_process_rss_mb: float,
        peak_system_used_mb: float,
        output: str,
        host: dict,
        ttft_s: float | None = None,
    ) -> RunResult:
        """Factory for a successful run."""
        tokens_per_s = (
            max_new_tokens / generate_time_s if generate_time_s and generate_time_s > 0 else None
        )
        return cls(
            backend=backend,
            model_id=model_id,
            prompt_chars=len(prompt),
            max_new_tokens=max_new_tokens,
            status=STATUS_SUCCESS,
            failure_reason=None,
            load_time_s=load_time_s,
            ttft_s=ttft_s,
            generate_time_s=generate_time_s,
            total_runtime_s=load_time_s + generate_time_s,
            tokens_per_s=tokens_per_s,
            peak_process_rss_mb=peak_process_rss_mb,
            peak_system_used_mb=peak_system_used_mb,
            output_preview=output[:200],
            host=host,
        )

    @classmethod
    def failed(
        cls,
        *,
        backend: str,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        reason: str,
        host: dict,
        load_time_s: float | None = None,
        peak_process_rss_mb: float | None = None,
        peak_system_used_mb: float | None = None,
    ) -> RunResult:
        """Factory for a failed run."""
        return cls(
            backend=backend,
            model_id=model_id,
            prompt_chars=len(prompt),
            max_new_tokens=max_new_tokens,
            status=STATUS_FAILED,
            failure_reason=reason,
            load_time_s=load_time_s,
            ttft_s=None,
            generate_time_s=None,
            total_runtime_s=load_time_s,
            tokens_per_s=None,
            peak_process_rss_mb=peak_process_rss_mb,
            peak_system_used_mb=peak_system_used_mb,
            output_preview=None,
            host=host,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return dataclasses.asdict(self)
