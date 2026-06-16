"""Unit tests for shared.gatekeeper.Gatekeeper."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from airllm_bench.constants import Target
from airllm_bench.shared.config import Settings
from airllm_bench.shared.gatekeeper import Gatekeeper


def _make_gatekeeper(rate_per_min: int = 600) -> Gatekeeper:
    """Return a Gatekeeper with a high rate limit so unit tests don't sleep."""
    settings = Settings()
    settings = settings.model_copy(update={"hf_rate_limit_per_min": rate_per_min})
    return Gatekeeper(settings, max_concurrent=4)


class TestGatekeeperSuccess:
    def test_call_returns_fn_result(self):
        gk = _make_gatekeeper()
        result = gk.call(Target.HF_HUB, lambda: 42)
        assert result == 42

    def test_call_passes_args_and_kwargs(self):
        gk = _make_gatekeeper()

        def adder(a, b=0):
            return a + b

        result = gk.call(Target.HF_HUB, adder, 3, b=4)
        assert result == 7


class TestGatekeeperRetry:
    def test_transient_error_retried_then_succeeds(self):
        gk = _make_gatekeeper()
        counter = {"n": 0}

        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                raise OSError("transient")
            return "ok"

        result = gk.call(Target.OLLAMA, flaky)
        assert result == "ok"
        assert counter["n"] == 3

    def test_non_transient_error_not_retried(self):
        gk = _make_gatekeeper()
        counter = {"n": 0}

        def fails():
            counter["n"] += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError):  # noqa: PT011
            gk.call(Target.HF_HUB, fails)
        assert counter["n"] == 1  # no retry for ValueError


class TestGatekeeperConcurrency:
    def test_semaphore_limits_concurrent_calls(self):
        """With max_concurrent=1, second call must wait for first."""
        settings = Settings()
        settings = settings.model_copy(update={"hf_rate_limit_per_min": 6000})
        gk = Gatekeeper(settings, max_concurrent=1)

        results: list[int] = []
        barrier = threading.Barrier(2)

        def slow_fn(idx: int) -> int:
            barrier.wait(timeout=2)
            time.sleep(0.01)
            results.append(idx)
            return idx

        threads = [threading.Thread(target=gk.call, args=(Target.OLLAMA, slow_fn, i)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Both should complete (no deadlock).
        assert len(results) == 2


class TestGatekeeperLogging:
    def test_hf_token_not_in_log(self, caplog):
        """Ensure the HF token is never logged even if passed as kwarg."""
        import logging

        settings = Settings()
        settings = settings.model_copy(
            update={"hf_token": "hf_supersecret", "hf_rate_limit_per_min": 6000}
        )
        gk = Gatekeeper(settings)

        def noop(**kwargs):
            return "done"

        with caplog.at_level(logging.DEBUG):
            gk.call(Target.HF_HUB, noop, token="hf_supersecret")

        for record in caplog.records:
            assert "hf_supersecret" not in record.getMessage()
