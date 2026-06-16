"""Application-wide configuration via pydantic-settings.

All tuneable values (model IDs, paths, rate limits, timeouts, etc.) are read
from environment variables or an optional .env file.  Hard-coded defaults exist
only as fallbacks that are safe for local development.

Secrets rule: HF_TOKEN is accepted but *never* printed, logged, or exposed in
any RunResult field.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration sourced from env / .env (in that priority order)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── HuggingFace ──────────────────────────────────────────────────────────
    hf_token: str = Field(default="", description="HuggingFace access token (gated models).")
    hf_rate_limit_per_min: int = Field(default=30, ge=1, description="Max HF Hub calls per minute.")

    # ── Models ───────────────────────────────────────────────────────────────
    model_id: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Primary large model ID on HuggingFace Hub.",
    )
    sanity_model_id: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Tiny model for pipeline sanity checks.",
    )
    ollama_sanity_model: str = Field(
        default="qwen2:0.5b",
        description="Ollama model tag for B1 sanity run.",
    )

    # ── Benchmark parameters ─────────────────────────────────────────────────
    prompt: str = Field(
        default="Explain virtual memory in one sentence.",
        description="Benchmark prompt sent to every backend.",
    )
    max_new_tokens: int = Field(default=16, ge=1, description="Tokens to generate per run.")
    seed: int = Field(default=42, description="Random seed for reproducible generation.")

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama REST API base URL.",
    )

    # ── AirLLM ───────────────────────────────────────────────────────────────
    shards_path: Path = Field(
        default=Path("data/shards"),
        description="Directory where AirLLM stores per-layer weight shards.",
    )

    # ── Safety / watchdog ────────────────────────────────────────────────────
    memory_ceiling_mb: float = Field(
        default=3500.0,
        ge=256.0,
        description="Abort run if process RSS exceeds this value (MB).",
    )
    run_timeout_s: float = Field(
        default=3600.0,
        ge=10.0,
        description="Hard per-run wall-time timeout (seconds).",
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    sample_interval_s: float = Field(
        default=0.25,
        gt=0.0,
        description="MemoryMonitor polling interval (seconds).",
    )

    # ── Output directories ───────────────────────────────────────────────────
    results_dir: Path = Field(default=Path("results"), description="Where JSON/CSV results go.")
    assets_dir: Path = Field(default=Path("assets"), description="Where chart PNGs go.")

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("model_id", "sanity_model_id", mode="before")
    @classmethod
    def _non_empty_string(cls, v: str) -> str:
        """Reject blank model IDs."""
        if not str(v).strip():
            msg = "model_id must not be blank"
            raise ValueError(msg)
        return v

    @field_validator("shards_path", "results_dir", "assets_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        """Accept string or Path for directory settings."""
        return Path(str(v))


def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Callers should use this factory rather than instantiating Settings directly
    so that the .env file is only parsed once.
    """
    return Settings()
