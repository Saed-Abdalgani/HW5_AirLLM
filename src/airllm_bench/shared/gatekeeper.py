"""Central API Gatekeeper — the single chokepoint for all external I/O.

Every call to HuggingFace Hub or Ollama REST **must** pass through this
Gatekeeper.  It enforces:

* **Retries** — exponential back-off via ``tenacity`` for transient errors.
* **Rate limiting** — minimum interval between successive calls to the same
  target, configurable from :class:`~airllm_bench.shared.config.Settings`.
* **Concurrency cap** — bounded ``threading.Semaphore`` prevents run-away
  parallel requests.
* **Structured logging** — records target, latency, and outcome; *never* logs
  tokens, headers, or secrets.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from airllm_bench.constants import Target
from airllm_bench.shared.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions considered transient — trigger a retry.
_TRANSIENT = (OSError, TimeoutError, ConnectionError)

_MAX_RETRIES = 4
_WAIT_MIN_S = 1.0
_WAIT_MAX_S = 30.0


class Gatekeeper:
    """Centralised external-call controller.

    Parameters
    ----------
    settings:
        Application settings.  Rate limits are read from here so they are
        never hard-coded.
    max_concurrent:
        Maximum number of concurrent external calls.  Callers that exceed this
        block until a slot is free.
    """

    def __init__(self, settings: Settings, *, max_concurrent: int = 4) -> None:
        self._settings = settings
        self._semaphore = threading.Semaphore(max_concurrent)
        # Per-target: timestamp of the last successful call.
        self._last_call: dict[Target, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, target: Target, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *fn* under retry, rate-limit, and concurrency policies.

        Parameters
        ----------
        target:
            Identifies the external system (used for rate-limit bookkeeping
            and logging).  Must be a :class:`~airllm_bench.constants.Target`
            enum member.
        fn:
            The callable to execute.
        *args, **kwargs:
            Forwarded to *fn*.

        Returns
        -------
        T
            Whatever *fn* returns.

        Raises
        ------
        Exception
            Any non-transient exception raised by *fn* propagates unchanged.
        """
        self._rate_limit(target)
        with self._semaphore:
            return self._call_with_retry(target, fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rate_limit(self, target: Target) -> None:
        """Block until the minimum inter-call interval has elapsed."""
        min_interval = 60.0 / self._settings.hf_rate_limit_per_min
        with self._lock:
            last = self._last_call.get(target, 0.0)
            elapsed = time.monotonic() - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_call[target] = time.monotonic()

    def _call_with_retry(
        self,
        target: Target,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Wrap *fn* with tenacity retry logic, logging each attempt."""

        @retry(
            retry=retry_if_exception_type(_TRANSIENT),
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=_WAIT_MIN_S, max=_WAIT_MAX_S),
            reraise=True,
        )
        def _inner() -> T:
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                latency = time.monotonic() - start
                logger.info(
                    "Gatekeeper OK",
                    extra={"target": target.value, "latency_s": round(latency, 3)},
                )
                return result
            except Exception as exc:
                latency = time.monotonic() - start
                logger.warning(
                    "Gatekeeper ERROR",
                    extra={
                        "target": target.value,
                        "latency_s": round(latency, 3),
                        "error": type(exc).__name__,
                    },
                )
                raise

        return _inner()
