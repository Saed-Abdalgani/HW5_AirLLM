"""Unit tests for metrics.Timer."""

from __future__ import annotations

import time

import pytest

from airllm_bench.services.metrics.timer import Timer


class TestTimerPhases:
    def test_load_phase_recorded(self):
        t = Timer()
        with t.phase("load"):
            time.sleep(0.01)
        assert t.load_time_s is not None
        assert t.load_time_s >= 0.005

    def test_generate_phase_recorded(self):
        t = Timer()
        with t.phase("generate"):
            time.sleep(0.01)
        assert t.generate_time_s is not None
        assert t.generate_time_s >= 0.005

    def test_total_is_sum_of_phases(self):
        t = Timer()
        with t.phase("load"):
            pass
        with t.phase("generate"):
            pass
        assert t.total_s is not None
        expected = (t.load_time_s or 0.0) + (t.generate_time_s or 0.0)
        assert abs(t.total_s - expected) < 1e-9

    def test_none_before_measurement(self):
        t = Timer()
        assert t.load_time_s is None
        assert t.generate_time_s is None
        assert t.total_s is None

    def test_get_arbitrary_phase(self):
        t = Timer()
        with t.phase("split"):
            pass
        assert t.get("split") is not None
        assert t.get("nonexistent") is None

    def test_exception_inside_phase_still_records(self):
        t = Timer()
        with pytest.raises(ValueError):  # noqa: PT011
            with t.phase("load"):
                raise ValueError("boom")
        # Time was still recorded even though an exception propagated.
        assert t.load_time_s is not None
