"""AirLLM backend (B3) — layer-by-layer fp16 inference on CPU.

AirLLM streams one transformer layer at a time from the per-layer shards on
disk, runs the forward pass, then discards the layer from RAM before loading
the next one.  Peak memory ≈ one layer instead of the full model — this is the
assignment's core claim.

Design notes (ADR-1):
- ``compression=None`` because bitsandbytes 4-bit requires CUDA, which is not
  available on this Intel/Windows host.
- ``device='cpu'`` — no CUDA.
- First call splits the checkpoint into per-layer shards (``layer_shards_saving_path``);
  subsequent calls reuse them.  Split time is recorded separately.
"""

from __future__ import annotations

import logging
from pathlib import Path

from airllm_bench.constants import BackendName
from airllm_bench.services.backends.base import InferenceBackend
from airllm_bench.shared.config import Settings

logger = logging.getLogger(__name__)


class AirllmBackend(InferenceBackend):
    """Inference backend using airllm.AutoModel for layer-by-layer execution.

    Parameters
    ----------
    settings:
        Application settings (model ID, shards path, token, etc.).
    model_id:
        Overrides ``settings.model_id`` if provided.
    """

    def __init__(self, settings: Settings, *, model_id: str | None = None) -> None:
        self._settings = settings
        self._model_id = model_id or settings.model_id
        self._shards_path = Path(settings.shards_path)
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:  # noqa: D102
        return BackendName.AIRLLM.value

    def load(self) -> None:
        """Initialise the AirLLM model.

        On first run, AirLLM will split the checkpoint into per-layer shards
        under ``shards_path``.  The shards are reused on subsequent runs.

        Raises
        ------
        RuntimeError
            If model initialisation fails.
        """
        from airllm import AutoModel
        from transformers import AutoTokenizer

        self._shards_path.mkdir(parents=True, exist_ok=True)
        hf_token = self._settings.hf_token or None

        logger.info("AirllmBackend: initialising %s …", self._model_id)
        self._model = AutoModel.from_pretrained(
            self._model_id,
            device="cpu",
            compression=None,
            layer_shards_saving_path=str(self._shards_path),
            hf_token=hf_token,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            token=hf_token,
        )
        logger.info("AirllmBackend: model ready (shards at %s).", self._shards_path)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Generate tokens layer-by-layer, returning the new text."""
        if self._model is None or self._tokenizer is None:
            msg = "Call load() before generate()."
            raise RuntimeError(msg)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        output = self._model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=False,
            return_dict_in_generate=True,
        )
        new_ids = output.sequences[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def teardown(self) -> None:  # noqa: D102
        del self._model
        self._model = None
        self._tokenizer = None
        logger.debug("AirllmBackend torn down.")
