"""Abstract base class for all inference backends.

Every concrete backend (Ollama, TransformersCPU, AirLLM, GPU importer) must
implement this interface.  The :class:`~airllm_bench.sdk.runner.BenchmarkRunner`
depends only on this ABC, keeping backends interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """Contract that each inference backend must fulfil."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (matches BackendName value)."""

    @abstractmethod
    def load(self) -> None:
        """Initialise the model / establish connection.

        Raises
        ------
        RuntimeError
            If the backend cannot be initialised (e.g. OOM, Ollama not running).
        """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Generate a completion for *prompt*.

        Parameters
        ----------
        prompt:
            Input text to complete.
        max_new_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        str
            The generated text (prompt excluded).

        Raises
        ------
        RuntimeError
            On generation failure.
        """

    def teardown(self) -> None:
        """Release resources (optional — override if needed)."""
