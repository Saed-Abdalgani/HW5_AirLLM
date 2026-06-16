"""TransformersCPU backend (B2) — full fp16 model loaded entirely into RAM.

This is the **baseline failure** backend.  For a model larger than available
free RAM (~2.5 GB on the target host), this backend is expected to raise
a ``MemoryError`` or ``RuntimeError`` during ``from_pretrained()``.  The
:class:`~airllm_bench.sdk.runner.BenchmarkRunner` catches that and records a
``RunResult(status="failed", failure_reason="OOM")``.

Precision: fp16 — same as AirLLM, so the comparison is apples-to-apples on
quality (unlike Ollama's q4).
"""

from __future__ import annotations

import logging

import torch

from airllm_bench.constants import BackendName
from airllm_bench.services.backends.base import InferenceBackend
from airllm_bench.shared.config import Settings

logger = logging.getLogger(__name__)


class TransformersCpuBackend(InferenceBackend):
    """Load a causal-LM entirely into CPU RAM using HuggingFace Transformers.

    Parameters
    ----------
    settings:
        Application settings.
    model_id:
        HuggingFace model ID to load (overrides ``settings.model_id``).
    """

    def __init__(self, settings: Settings, *, model_id: str | None = None) -> None:
        self._settings = settings
        self._model_id = model_id or settings.model_id
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:  # noqa: D102
        return BackendName.TRANSFORMERS_CPU.value

    def load(self) -> None:
        """Load model + tokenizer into CPU RAM (fp16).

        Raises
        ------
        MemoryError | RuntimeError
            If the model does not fit in available RAM.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("TransformersCpuBackend: loading %s …", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        self._model.eval()
        logger.info("TransformersCpuBackend: model loaded.")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Run greedy generation for *max_new_tokens* tokens."""
        if self._model is None or self._tokenizer is None:
            msg = "Call load() before generate()."
            raise RuntimeError(msg)
        torch.manual_seed(self._settings.seed)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Strip the prompt tokens from the output.
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def teardown(self) -> None:  # noqa: D102
        del self._model
        self._model = None
        self._tokenizer = None
        logger.debug("TransformersCpuBackend torn down; model freed.")
