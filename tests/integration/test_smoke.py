"""Integration tests — require live services (Ollama, HuggingFace, real models).

Run explicitly with::

    uv run pytest -m integration -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from airllm_bench.constants import BackendName
from airllm_bench.sdk.runner import BenchmarkRunner
from airllm_bench.shared.config import Settings


def _ollama_available() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.mark.integration
def test_ollama_api_reachable():
    """Verify the Ollama REST API responds when the daemon is running."""
    if not _ollama_available():
        pytest.skip("Ollama not running on localhost:11434")
    tags = requests.get("http://localhost:11434/api/tags", timeout=3).json()
    assert "models" in tags


@pytest.mark.integration
def test_ollama_backend_end_to_end(tmp_path: Path):
    """Run the Ollama sanity backend against a live daemon."""
    if not _ollama_available():
        pytest.skip("Ollama not running on localhost:11434")

    settings = Settings(
        results_dir=str(tmp_path),
        max_new_tokens=4,
    )
    runner = BenchmarkRunner(settings)
    result = runner.run(BackendName.OLLAMA)

    assert result.status in {"success", "failed"}
    json_files = list(tmp_path.glob("run_ollama_*.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["backend"] == "ollama"
