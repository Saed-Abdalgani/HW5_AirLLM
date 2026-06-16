"""Ollama backend (B1) — REST calls via the Gatekeeper.

Talks to a locally-running Ollama daemon at the configured base URL.
Ollama handles its own quantisation (q4 GGUF), so no torch is needed here.

Precision note (documented in results/analysis.md):
Ollama uses 4-bit quantised GGUF weights, while TransformersCPU and AirLLM
use fp16.  The headline comparison therefore emphasises *feasibility* and
*peak memory*, with latency presented in context.
"""

from __future__ import annotations

import logging

import requests

from airllm_bench.constants import BackendName, Target
from airllm_bench.services.backends.base import InferenceBackend
from airllm_bench.shared.config import Settings
from airllm_bench.shared.gatekeeper import Gatekeeper

logger = logging.getLogger(__name__)


class OllamaBackend(InferenceBackend):
    """Inference backend that delegates to a local Ollama server via REST.

    Parameters
    ----------
    settings:
        Application settings (Ollama URL, model tag, etc.).
    gatekeeper:
        Central call controller for rate-limiting and retries.
    model_tag:
        Ollama model tag to use (defaults to ``settings.ollama_sanity_model``).
    """

    def __init__(
        self,
        settings: Settings,
        gatekeeper: Gatekeeper,
        *,
        model_tag: str | None = None,
    ) -> None:
        self._settings = settings
        self._gatekeeper = gatekeeper
        self._model_tag = model_tag or settings.ollama_sanity_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._loaded = False

    @property
    def name(self) -> str:  # noqa: D102
        return BackendName.OLLAMA.value

    def load(self) -> None:
        """Verify the Ollama daemon is reachable and the model is available."""
        url = f"{self._base_url}/api/tags"
        resp = self._gatekeeper.call(Target.OLLAMA, requests.get, url, timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(self._model_tag in m for m in models):
            logger.warning(
                "Model %s not found in Ollama; run `ollama pull %s` first.",
                self._model_tag,
                self._model_tag,
            )
        self._loaded = True
        logger.info("OllamaBackend loaded model=%s", self._model_tag)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Send a generate request to Ollama and return the response text."""
        if not self._loaded:
            msg = "Call load() before generate()."
            raise RuntimeError(msg)
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }
        resp = self._gatekeeper.call(
            Target.OLLAMA,
            requests.post,
            url,
            json=payload,
            timeout=self._settings.run_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", ""))

    def teardown(self) -> None:  # noqa: D102
        self._loaded = False
        logger.debug("OllamaBackend torn down.")
