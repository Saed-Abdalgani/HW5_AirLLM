"""Wall-clock timer for benchmark phases.

Usage::

    timer = Timer()
    with timer.phase("load"):
        backend.load()
    with timer.phase("generate"):
        text = backend.generate(prompt, max_new_tokens)

    print(timer.load_time_s, timer.generate_time_s, timer.total_s)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


class Timer:
    """Records wall-clock durations for named benchmark phases.

    Supported phases: ``"load"`` and ``"generate"``.  Other phase names are
    stored in ``extra`` for forward-compatibility.
    """

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Context manager that measures the wall-clock duration of a phase."""
        start = time.monotonic()
        try:
            yield
        finally:
            self._phases[name] = time.monotonic() - start

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def load_time_s(self) -> float | None:
        """Seconds spent in the ``"load"`` phase, or ``None`` if not measured."""
        return self._phases.get("load")

    @property
    def generate_time_s(self) -> float | None:
        """Seconds spent in the ``"generate"`` phase, or ``None`` if not measured."""
        return self._phases.get("generate")

    @property
    def total_s(self) -> float | None:
        """Sum of all measured phases, or ``None`` if nothing was measured."""
        if not self._phases:
            return None
        return sum(self._phases.values())

    def get(self, name: str) -> float | None:
        """Return the duration for an arbitrary phase name."""
        return self._phases.get(name)
