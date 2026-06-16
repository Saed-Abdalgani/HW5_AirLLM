"""Model provider — resolve a model ID to a local path via the Gatekeeper.

Performs a pre-flight disk-space check before attempting a download, and
delegates the actual download to ``huggingface_hub.snapshot_download`` through
the Gatekeeper (so retries and rate-limiting apply).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from airllm_bench.constants import REASON_DISK, Target
from airllm_bench.shared.config import Settings
from airllm_bench.shared.gatekeeper import Gatekeeper

logger = logging.getLogger(__name__)

# Estimated on-disk sizes (bytes) for common model classes.
# Used for the pre-flight disk check when the exact size is unknown.
_SIZE_ESTIMATES: dict[str, int] = {
    "0.5b": 1_000_000_000,
    "1.5b": 3_000_000_000,
    "3b": 6_500_000_000,
    "7b": 14_000_000_000,
    "8b": 16_000_000_000,
}

_SAFETY_FACTOR = 1.5  # require 1.5× estimated size free on disk


class ModelProvider:
    """Resolve and download HuggingFace models via the Gatekeeper.

    Parameters
    ----------
    settings:
        Application settings (HF token, rate limit, etc.).
    gatekeeper:
        Central call controller.
    """

    def __init__(self, settings: Settings, gatekeeper: Gatekeeper) -> None:
        self._settings = settings
        self._gatekeeper = gatekeeper

    def ensure(self, model_id: str) -> Path:
        """Return the local path to *model_id*, downloading if necessary.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier (e.g. ``Qwen/Qwen2.5-3B-Instruct``).

        Returns
        -------
        Path
            Local directory containing the model weights.

        Raises
        ------
        OSError
            If disk space is insufficient.
        RuntimeError
            If the download fails after retries.
        """
        self._check_disk(model_id)
        token = self._settings.hf_token or None
        logger.info("ModelProvider: ensuring %s is cached locally …", model_id)

        from huggingface_hub import snapshot_download

        local_path = self._gatekeeper.call(
            Target.HF_HUB,
            snapshot_download,
            model_id,
            token=token,
        )
        logger.info("ModelProvider: %s available at %s", model_id, local_path)
        return Path(local_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_disk(self, model_id: str) -> None:
        """Raise OSError if free disk space is below the estimated requirement."""
        estimate = self._estimate_size(model_id)
        required = int(estimate * _SAFETY_FACTOR)
        free = shutil.disk_usage("/").free
        if free < required:
            msg = (
                f"{REASON_DISK}: need ~{required // 1_000_000_000} GB free for {model_id}, "
                f"only {free // 1_000_000_000} GB available."
            )
            raise OSError(msg)

    @staticmethod
    def _estimate_size(model_id: str) -> int:
        """Return a rough byte estimate for a model ID, based on known sizes."""
        lower = model_id.lower()
        for tag, size in _SIZE_ESTIMATES.items():
            if tag in lower:
                return size
        # Unknown — assume 7B to be safe.
        return _SIZE_ESTIMATES["7b"]
