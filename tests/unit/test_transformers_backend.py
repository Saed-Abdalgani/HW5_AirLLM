"""Unit tests for TransformersCpuBackend — mocked model/tokenizer.

All heavy torch/transformers imports are mocked so no model weights are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from airllm_bench.services.backends.transformers_cpu_backend import TransformersCpuBackend
from airllm_bench.shared.config import Settings


def _make_backend(model_id: str = "test/tiny-model") -> TransformersCpuBackend:
    settings = Settings()
    return TransformersCpuBackend(settings, model_id=model_id)


class TestTransformersCpuBackendLoad:
    def test_load_calls_from_pretrained(self):
        backend = _make_backend()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as tok_patch,
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mdl_patch,
        ):
            backend.load()

        tok_patch.assert_called_once_with("test/tiny-model")
        mdl_patch.assert_called_once()
        # Verify torch_dtype and device_map kwargs
        call_kwargs = mdl_patch.call_args.kwargs
        assert call_kwargs.get("torch_dtype") == torch.float16
        assert call_kwargs.get("device_map") == "cpu"

    def test_load_sets_model_and_tokenizer(self):
        backend = _make_backend()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
        ):
            backend.load()

        assert backend._model is not None  # noqa: SLF001
        assert backend._tokenizer is not None  # noqa: SLF001


class TestTransformersCpuBackendGenerate:
    def _loaded_backend(self):
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

        # Tokenizer encodes to a 2-token prompt
        prompt_ids = torch.tensor([[1, 2]])
        backend._tokenizer.return_value = {"input_ids": prompt_ids}  # noqa: SLF001
        # Model generates 4 tokens (2 prompt + 2 new)
        generated_ids = torch.tensor([[1, 2, 3, 4]])
        backend._model.generate.return_value = generated_ids  # noqa: SLF001
        backend._tokenizer.decode.return_value = "generated text"  # noqa: SLF001

        result = backend.generate("Hello world", 2)
        assert result == "generated text"

    def test_teardown_clears_model(self):
        backend = self._loaded_backend()
        backend.teardown()
        assert backend._model is None  # noqa: SLF001
        assert backend._tokenizer is None  # noqa: SLF001

    def test_name_property(self):
        backend = _make_backend()
        assert backend.name == "transformers-cpu"
