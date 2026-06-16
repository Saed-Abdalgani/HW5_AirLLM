"""_executor — low-level run orchestration for BenchmarkRunner.

Extracted from ``runner.py`` to keep each file under 150 lines.
Contains the ``execute_run`` function that wires timer + memory monitor +
backend and maps every exception class to a ``RunResult``.
"""

from __future__ import annotations

import logging

from airllm_bench.constants import REASON_OOM, REASON_TIMEOUT, REASON_UNKNOWN
from airllm_bench.services.metrics.memory_monitor import MemoryMonitor
from airllm_bench.services.metrics.run_result import RunResult
from airllm_bench.services.metrics.timer import Timer
from airllm_bench.shared.config import Settings

logger = logging.getLogger(__name__)


def execute_run(
    backend,  # noqa: ANN001  (InferenceBackend — avoids circular import)
    settings: Settings,
    monitor: MemoryMonitor,
    timer: Timer,
    host: dict,
) -> RunResult:
    """Orchestrate load → generate, capturing metrics and mapping all errors.

    Parameters
    ----------
    backend:
        A concrete :class:`~airllm_bench.services.backends.base.InferenceBackend`.
    settings:
        Application settings (prompt, max_new_tokens, model_id).
    monitor:
        Already-constructed (but not yet started) :class:`MemoryMonitor`.
    timer:
        Fresh :class:`Timer` instance.
    host:
        Host spec dict to embed in the result.

    Returns
    -------
    RunResult
        Always returns — never raises.  Errors are captured as failed results.
    """
    prompt = settings.prompt
    max_new_tokens = settings.max_new_tokens

    monitor.start()
    try:
        with timer.phase("load"):
            backend.load()

        with timer.phase("generate"):
            output = backend.generate(prompt, max_new_tokens)

        monitor.stop()
        return RunResult.success(
            backend=backend.name,
            model_id=settings.model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            load_time_s=timer.load_time_s or 0.0,
            generate_time_s=timer.generate_time_s or 0.0,
            peak_process_rss_mb=monitor.peak_process_rss_mb,
            peak_system_used_mb=monitor.peak_system_used_mb,
            output=output,
            host=host,
        )

    except (MemoryError, RuntimeError) as exc:
        monitor.stop()
        reason = REASON_OOM if "memory" in str(exc).lower() else REASON_UNKNOWN
        logger.error("Backend %s failed: %s", backend.name, exc)
        return _failed(backend, settings, prompt, max_new_tokens, reason, timer, monitor, host)

    except TimeoutError:
        monitor.stop()
        logger.error("Backend %s timed out.", backend.name)
        return _failed(
            backend, settings, prompt, max_new_tokens, REASON_TIMEOUT, timer, monitor, host
        )

    except Exception as exc:  # noqa: BLE001
        monitor.stop()
        logger.exception("Backend %s crashed unexpectedly.", backend.name)
        return _failed(
            backend,
            settings,
            prompt,
            max_new_tokens,
            f"{type(exc).__name__}: {exc}",
            timer,
            monitor,
            host,
        )

    finally:
        backend.teardown()


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _failed(
    backend,  # noqa: ANN001
    settings: Settings,
    prompt: str,
    max_new_tokens: int,
    reason: str,
    timer: Timer,
    monitor: MemoryMonitor,
    host: dict,
) -> RunResult:
    """Build a failed RunResult from partial timing/memory data."""
    return RunResult.failed(
        backend=backend.name,
        model_id=settings.model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        reason=reason,
        host=host,
        load_time_s=timer.load_time_s,
        peak_process_rss_mb=monitor.peak_process_rss_mb,
        peak_system_used_mb=monitor.peak_system_used_mb,
    )
