"""Host specification helper — captures CPU/RAM/OS/Python metadata.

The ``host_spec`` dict is embedded in every ``RunResult.host`` field, and also
persisted as ``results/host_spec.json`` for reproducibility audits.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import psutil


def capture_host_spec() -> dict:
    """Return a dictionary describing the current host hardware and software.

    Returns
    -------
    dict
        Keys: ``cpu``, ``cpu_count_logical``, ``cpu_count_physical``,
        ``total_ram_mb``, ``free_ram_mb``, ``os``, ``python``, ``has_cuda``.
    """
    mem = psutil.virtual_memory()
    mb = 1024.0 * 1024.0

    # Attempt to detect CUDA — graceful fallback if torch not installed.
    has_cuda = False
    try:
        import torch  # noqa: PLC0415

        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_mb": round(mem.total / mb, 1),
        "free_ram_mb": round(mem.available / mb, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version,
        "has_cuda": has_cuda,
    }


def write_host_spec(results_dir: Path) -> Path:
    """Capture host spec and write it to ``results_dir/host_spec.json``.

    Parameters
    ----------
    results_dir:
        Directory where the file will be written (created if absent).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    spec = capture_host_spec()
    out = results_dir / "host_spec.json"
    out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return out
