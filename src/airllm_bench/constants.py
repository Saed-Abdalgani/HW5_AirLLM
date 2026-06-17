"""Project-wide constants and enumerations.

All symbolic names (backend identifiers, unit conversions, file-name templates,
enum members) live here so they are never duplicated across modules.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

MB: float = 1024.0 * 1024.0  # bytes → megabytes divisor
GB: float = MB * 1024.0


# ---------------------------------------------------------------------------
# Backend identifiers
# ---------------------------------------------------------------------------


class BackendName(StrEnum):
    """Identifies the inference backend used for a benchmark run."""

    OLLAMA = "ollama"
    TRANSFORMERS_CPU = "transformers-cpu"
    AIRLLM = "airllm"
    GPU = "gpu"


# ---------------------------------------------------------------------------
# Gatekeeper targets
# ---------------------------------------------------------------------------


class Target(StrEnum):
    """External systems routed through the Gatekeeper."""

    HF_HUB = "hf_hub"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# File-name templates  (backend, timestamp as ISO-8601 compact string)
# ---------------------------------------------------------------------------

RESULT_FILENAME_TEMPLATE = "run_{backend}_{timestamp}.json"
HOST_SPEC_FILENAME = "host_spec.json"
COMPARISON_CSV_FILENAME = "comparison.csv"

# ---------------------------------------------------------------------------
# Run-result status values
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_NA = "n/a"

# ---------------------------------------------------------------------------
# Failure reason labels
# ---------------------------------------------------------------------------

REASON_OOM = "OOM"
REASON_TIMEOUT = "timeout"
REASON_DISK = "insufficient_disk"
REASON_UNKNOWN = "unknown"
