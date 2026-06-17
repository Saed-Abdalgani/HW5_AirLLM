"""Unit tests for ModelProvider — disk check and mocked download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from airllm_bench.services.model_provider import ModelProvider
from airllm_bench.shared.config import Settings


def _make_provider(free_bytes: int = 100 * 10**9) -> ModelProvider:
    settings = Settings()
    gatekeeper = MagicMock()
    # Pass-through: gatekeeper.call just calls the function.
    gatekeeper.call.side_effect = lambda target, fn, *a, **kw: fn(*a, **kw)

    with patch("shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=free_bytes)
        provider = ModelProvider(settings, gatekeeper)
        provider._check_disk = lambda model_id: None  # noqa: SLF001  # patch out for unit
    return provider


class TestModelProvider:
    def test_ensure_calls_snapshot_download(self):
        provider = _make_provider()

        fake_path = "/fake/model/path"
        with patch(
            "huggingface_hub.snapshot_download",
            return_value=fake_path,
        ) as mock_dl:
            result = provider.ensure("Qwen/Qwen2.5-0.5B-Instruct")

        mock_dl.assert_called_once()
        assert result == Path(fake_path)

    def test_insufficient_disk_raises(self):
        settings = Settings()
        gatekeeper = MagicMock()
        provider = ModelProvider(settings, gatekeeper)

        with (
            patch("shutil.disk_usage") as mock_du,
            pytest.raises(OSError, match="insufficient_disk"),
        ):
            mock_du.return_value = MagicMock(free=1_000_000)  # 1 MB — way too small
            provider._check_disk("Qwen/Qwen2.5-3B-Instruct")  # noqa: SLF001

    def test_estimate_size_for_known_model(self):
        estimate = ModelProvider._estimate_size("Qwen/Qwen2.5-0.5B-Instruct")  # noqa: SLF001
        assert estimate == 1_000_000_000

    def test_estimate_size_unknown_defaults_to_7b(self):
        estimate = ModelProvider._estimate_size("unknown/mystery-model")  # noqa: SLF001
        assert estimate == 14_000_000_000
