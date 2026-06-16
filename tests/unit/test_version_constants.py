"""Unit tests for shared/version.py and constants.py."""

from __future__ import annotations

from airllm_bench.constants import BackendName, Target
from airllm_bench.shared.version import __version__


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestBackendName:
    def test_all_values_present(self):
        values = {b.value for b in BackendName}
        assert "ollama" in values
        assert "transformers-cpu" in values
        assert "airllm" in values
        assert "gpu" in values

    def test_is_str_enum(self):
        assert isinstance(BackendName.OLLAMA, str)
        assert BackendName.OLLAMA == "ollama"


class TestTarget:
    def test_hf_hub_and_ollama(self):
        assert Target.HF_HUB.value == "hf_hub"
        assert Target.OLLAMA.value == "ollama"

    def test_is_str_enum(self):
        assert isinstance(Target.HF_HUB, str)
