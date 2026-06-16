"""Unit tests for OllamaBackend — mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from airllm_bench.services.backends.ollama_backend import OllamaBackend
from airllm_bench.shared.config import Settings


def _make_backend() -> OllamaBackend:
    settings = Settings()
    gatekeeper = MagicMock()
    # Make the gatekeeper.call execute the function directly (no rate-limit in tests).
    gatekeeper.call.side_effect = lambda target, fn, *a, **kw: fn(*a, **kw)
    return OllamaBackend(settings, gatekeeper, model_tag="qwen2:0.5b")


class TestOllamaBackendLoad:
    def test_load_succeeds_when_model_listed(self):
        backend = _make_backend()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen2:0.5b"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            backend.load()

        assert backend._loaded  # noqa: SLF001

    def test_load_warns_when_model_absent(self, caplog):
        backend = _make_backend()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "other:model"}]}
        mock_resp.raise_for_status = MagicMock()

        import logging

        with patch("requests.get", return_value=mock_resp), caplog.at_level(logging.WARNING):
            backend.load()
        assert "not found" in caplog.text.lower()

    def test_load_raises_on_http_error(self):
        backend = _make_backend()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")

        with patch("requests.get", return_value=mock_resp), pytest.raises(requests.HTTPError):
            backend.load()


class TestOllamaBackendGenerate:
    def test_generate_returns_response_text(self):
        backend = _make_backend()
        backend._loaded = True  # noqa: SLF001

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Virtual memory is a management technique."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            text = backend.generate("Explain virtual memory.", 16)

        assert "Virtual memory" in text

    def test_generate_raises_if_not_loaded(self):
        backend = _make_backend()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            backend.generate("prompt", 8)

    def test_generate_raises_on_http_error(self):
        backend = _make_backend()
        backend._loaded = True  # noqa: SLF001
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")

        with patch("requests.post", return_value=mock_resp), pytest.raises(requests.HTTPError):
            backend.generate("prompt", 8)

    def test_teardown_resets_loaded_flag(self):
        backend = _make_backend()
        backend._loaded = True  # noqa: SLF001
        backend.teardown()
        assert not backend._loaded  # noqa: SLF001
