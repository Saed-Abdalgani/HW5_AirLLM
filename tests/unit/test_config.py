"""Unit tests for shared/config.py (Settings)."""

from __future__ import annotations

import pytest

from airllm_bench.shared.config import Settings


class TestSettingsDefaults:
    """Verify that all default values load without a .env file."""

    def test_default_model_id(self):
        s = Settings()
        assert "Qwen" in s.model_id or "meta-llama" in s.model_id

    def test_default_max_new_tokens(self):
        s = Settings()
        assert s.max_new_tokens == 16

    def test_default_memory_ceiling(self):
        s = Settings()
        assert s.memory_ceiling_mb == 3500.0

    def test_default_sample_interval(self):
        s = Settings()
        assert s.sample_interval_s == 0.25

    def test_default_seed(self):
        s = Settings()
        assert s.seed == 42

    def test_hf_token_empty_by_default(self):
        """HF token defaults to empty string — never raises."""
        s = Settings()
        assert s.hf_token == ""

    def test_paths_are_path_objects(self):
        from pathlib import Path

        s = Settings()
        assert isinstance(s.results_dir, Path)
        assert isinstance(s.assets_dir, Path)
        assert isinstance(s.shards_path, Path)


class TestSettingsEnvOverride:
    """Verify environment-variable override behaviour."""

    def test_override_max_new_tokens(self, monkeypatch):
        monkeypatch.setenv("MAX_NEW_TOKENS", "32")
        s = Settings()
        assert s.max_new_tokens == 32

    def test_override_model_id(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
        s = Settings()
        assert s.model_id == "meta-llama/Llama-3.2-3B-Instruct"

    def test_override_ollama_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
        s = Settings()
        assert s.ollama_base_url == "http://localhost:9999"

    def test_string_path_coerced_to_path(self, monkeypatch):
        from pathlib import Path

        monkeypatch.setenv("RESULTS_DIR", "my_results")
        s = Settings()
        assert s.results_dir == Path("my_results")


class TestSettingsValidation:
    """Verify that invalid inputs raise validation errors."""

    def test_blank_model_id_raises(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("MODEL_ID", "   ")
        with pytest.raises(ValidationError):
            Settings()

    def test_negative_max_tokens_raises(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("MAX_NEW_TOKENS", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_low_memory_ceiling_raises(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("MEMORY_CEILING_MB", "100")
        with pytest.raises(ValidationError):
            Settings()

    def test_zero_sample_interval_raises(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("SAMPLE_INTERVAL_S", "0")
        with pytest.raises(ValidationError):
            Settings()
