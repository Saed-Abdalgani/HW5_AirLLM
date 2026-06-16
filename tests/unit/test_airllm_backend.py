"""Unit tests for AirllmBackend — mocked airllm.AutoModel and transformers.

No real model downloads or GPU needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from airllm_bench.services.backends.airllm_backend import AirllmBackend
from airllm_bench.shared.config import Settings


def _make_backend(model_id: str = "test/model") -> AirllmBackend:
    settings = Settings()
    return AirllmBackend(settings, model_id=model_id)


class TestAirllmBackendLoad:
    def test_load_calls_automodel_with_correct_kwargs(self):
        backend = _make_backend()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("airllm.AutoModel.from_pretrained", return_value=mock_model) as am_patch,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            backend.load()

        call_kwargs = am_patch.call_args.kwargs
        assert call_kwargs.get("device") == "cpu"
        assert call_kwargs.get("compression") is None
        assert backend._model is mock_model  # noqa: SLF001

    def test_load_creates_shards_dir(self, tmp_path):
        settings = Settings()
        # Override shards_path to a temp dir
        backend = AirllmBackend(settings, model_id="test/model")
        backend._shards_path = tmp_path / "shards"  # noqa: SLF001

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("airllm.AutoModel.from_pretrained", return_value=mock_model),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            backend.load()

        assert backend._shards_path.exists()  # noqa: SLF001


class TestAirllmBackendGenerate:
    def _loaded_backend(self) -> AirllmBackend:
        backend = _make_backend()
        backend._model = MagicMock()  # noqa: SLF001
        backend._tokenizer = MagicMock()  # noqa: SLF001
        return backend

    def test_generate_raises_if_not_loaded(self):
        backend = _make_backend()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            backend.generate("prompt", 8)

    def test_generate_returns_decoded_text(self):
        backend = self._loaded_backend()

        prompt_ids = torch.tensor([[1, 2]])
        backend._tokenizer.return_value = {"input_ids": prompt_ids}  # noqa: SLF001

        output_mock = MagicMock()
        output_mock.sequences = [torch.tensor([1, 2, 10, 11])]
        backend._model.generate.return_value = output_mock  # noqa: SLF001
        backend._tokenizer.decode.return_value = "layer streamed text"  # noqa: SLF001

        result = backend.generate("Explain virtual memory", 2)
        assert result == "layer streamed text"

    def test_teardown_clears_model(self):
        backend = self._loaded_backend()
        backend.teardown()
        assert backend._model is None  # noqa: SLF001
        assert backend._tokenizer is None  # noqa: SLF001

    def test_name_property(self):
        backend = _make_backend()
        assert backend.name == "airllm"
