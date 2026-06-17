# airllm-bench

> **HW5 — Proving AirLLM Value on a Memory-Constrained Machine**  
> Course: AI Agents Orchestration · Project codename: `airllm-bench`

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Project Does

This benchmark harness **empirically proves** that [AirLLM](https://github.com/lyogavin/airllm)
enables running a model that otherwise does **not fit in available RAM**, on a GPU-less,
memory-constrained laptop.

It runs the *same model* through multiple execution backends and measures:

| Metric | Captured |
|--------|----------|
| Load time | ✅ |
| Generate time | ✅ |
| Total runtime | ✅ |
| Tokens per second | ✅ |
| Peak process RSS (MB) | ✅ |
| Peak system memory (MB) | ✅ |
| Success / failure reason | ✅ |

### The Three Outcomes (Core Assignment)

| Backend | Model | Expected Result |
|---------|-------|-----------------|
| `transformers-cpu` | Qwen2.5-3B-Instruct | ❌ **OOM** — ~3.49 GB RSS at crash |
| `airllm` | Qwen2.5-3B-Instruct | ✅ **Success** — peak RSS ~2.94 GB, high latency |
| `ollama` | qwen2:0.5b | ✅ Sanity pass — 3.8 tok/s |

> **AirLLM works because it streams one transformer layer at a time from disk**
> (≈ explicit paging of model weights), keeping peak memory ≈ one layer instead
> of the entire model. The baseline loads everything at once → OOM.

---

## 🏗️ Architecture

```
CLI / Notebooks
      │  (SDK only — no direct service imports)
      ▼
sdk.BenchmarkRunner ──── sdk.summarize / sdk.plot_comparison
      │
      ├── services.backends.OllamaBackend          (B1 — Ollama REST)
      ├── services.backends.TransformersCpuBackend (B2 — baseline fail)
      ├── services.backends.AirllmBackend          (B3 — layer streaming)
      └── services.backends.GpuResultImporter      (B4 — Colab JSON)
      │
      ├── services.metrics.Timer
      ├── services.metrics.MemoryMonitor
      └── services.metrics.MetricsRecorder
      │
      └── shared.gatekeeper.Gatekeeper ──► HuggingFace Hub
                                       └─► Ollama REST
```

All external calls (HF Hub download, Ollama REST) pass through the **Gatekeeper**,
which enforces retries, rate-limits, backpressure, and structured logging — with
**no secrets ever logged**.

---

## 📦 Installation

### Prerequisites

- **Python 3.11** (exactly — pinned for package compatibility)
- **[uv](https://docs.astral.sh/uv/)** — `pip install uv` or `winget install astral-sh.uv`
- **[Ollama](https://ollama.ai/)** for the B1 backend (separate installer)

### Steps

```powershell
# 1. Clone / enter the repo
cd HW5_AirLLM

# 2. Install all dependencies (CPU-only torch, no CUDA bloat)
uv sync

# 3. Copy the env example and fill in your HuggingFace token
copy .env-example .env
# Edit .env: set HF_TOKEN=hf_your_token_here  (needed for gated models)

# 4. Verify the install
uv run python -c "import airllm_bench; print(airllm_bench.__version__)"
# → 0.1.0
```

### Ollama Setup (for B1 backend)

```powershell
# Install Ollama (Windows)
winget install Ollama.Ollama

# Pull the sanity model
ollama pull qwen2:0.5b

# Verify it works
ollama run qwen2:0.5b "hi"

# Check the API is up
Invoke-RestMethod http://localhost:11434/api/tags
```

---

## 🚀 Usage

All commands use the `airllm-bench` CLI entry point.

### Print host hardware spec

```powershell
uv run airllm-bench host-spec
```

### Run a single backend

```powershell
# Ollama sanity check (tiny model, fast)
uv run airllm-bench run --backend ollama

# Baseline failure (large model → OOM)
uv run airllm-bench run --backend transformers-cpu

# AirLLM success (same large model, layer-by-layer)
uv run airllm-bench run --backend airllm
```

### Run all backends sequentially

```powershell
uv run airllm-bench run-all
```

### Print comparison report + generate charts

```powershell
uv run airllm-bench report
# Charts saved to: assets/
```

---

## ⚙️ Configuration

All tuneable values live in `.env` (copy from `.env-example`):

| Key | Default | Description |
|-----|---------|-------------|
| `HF_TOKEN` | *(empty)* | HuggingFace token — required for gated models |
| `MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Primary benchmark model |
| `SANITY_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Tiny model for sanity runs |
| `OLLAMA_SANITY_MODEL` | `qwen2:0.5b` | Ollama model tag |
| `PROMPT` | `Explain virtual memory in one sentence.` | Benchmark prompt |
| `MAX_NEW_TOKENS` | `16` | Tokens to generate (keep tiny on slow CPU) |
| `MEMORY_CEILING_MB` | `3500` | RSS watchdog abort threshold |
| `RUN_TIMEOUT_S` | `3600` | Per-run hard timeout |
| `SHARDS_PATH` | `data/shards` | AirLLM per-layer shard directory |
| `RESULTS_DIR` | `results` | JSON/CSV output directory |
| `ASSETS_DIR` | `assets` | Chart PNG output directory |

**Secrets rule:** `HF_TOKEN` is read from env only — never logged, never committed.

---

## 🧪 Testing

```powershell
# Unit tests only (default — integration tests excluded)
uv run pytest

# Explicit unit-test path
uv run pytest tests/unit/ -v

# Coverage gate (≥ 85%, integration excluded)
uv run pytest -m "not integration" --cov=airllm_bench --cov-report=term-missing --cov-fail-under=85

# Lint check (must be zero violations)
uv run ruff check .

# Integration tests (requires Ollama running + network)
uv run pytest -m integration -v
```

**Coverage target:** ≥ 85% on unit tests with mocked externals (HF Hub, Ollama HTTP,
`airllm.AutoModel`, torch loaders). Heavy real-model runs are tagged
`@pytest.mark.integration` and excluded from the default `pytest` invocation.

---

## 📊 Results

Results from the target host: My PC only has 8 GB RAM. 5 GB is used by the OS by default, and ~1 GB is used for running Python files. This leaves only ~2 GB of free RAM for this task, which is why we used the `Qwen/Qwen2.5-3B-Instruct` model to validate the constraint.

| Backend | Model | Status | Response Time (s) | Peak Memory (MB) | Total Runtime (s) | Tokens/s |
|---------|-------|--------|-------------------|------------------|-------------------|----------|
| GPU | Qwen2.5-3B-Instruct | success | 2.67 | 1894.2 | 4.86 | 5.984 |
| CPU | Qwen2.5-3B-Instruct | failed (OOM) | — | 3487.9 | 47.31 | — |
| AirLLM | Qwen2.5-3B-Instruct | success | 1847.52 | 2941.7 | 2160.36 | 0.009 |
| Ollama | qwen2:0.5b | success | 4.21 | 612.4 | 7.06 | 3.797 |

> **Key finding:** AirLLM completed the 3B model where the all-in-RAM CPU baseline OOM'd,
> trading ~36 minutes of runtime for a 546 MB lower peak RSS via layer-by-layer streaming.
> GPU (Colab T4) is the speed reference; Ollama uses a smaller q4-quantised model.

Charts in [`assets/`](assets/).  Full table in [`results/comparison_table.md`](results/comparison_table.md).
Raw JSON + CSV in [`results/`](results/).

---

## 🔍 Troubleshooting

### `uv sync` fails
- Ensure Python 3.11 is installed: `uv python install 3.11`
- CPU-only torch is indexed separately — `uv` handles this via `pyproject.toml`.

### AirLLM OOM even with layer streaming
- Close other apps before the run to free RAM.
- If still OOM on 3B, fall back: set `MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct` in `.env`.
- Check `MEMORY_CEILING_MB` — raise pagefile size on Windows if needed.

### Ollama not found / connection refused
- Start Ollama: open **Ollama** from the system tray / run `ollama serve`.
- Pull the model: `ollama pull qwen2:0.5b`.

### AirLLM run takes very long
- Expected! On a 2-core CPU, 16 tokens via layer-by-layer fp16 takes 20–40 min.
- This is the documented latency trade-off. Reduce `MAX_NEW_TOKENS` (min: 1).

### Long Windows path errors for shards
- AirLLM creates many per-layer files. Enable long paths:
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`.

---

## 📁 Project Structure

```
HW5_AirLLM/
├── src/airllm_bench/
│   ├── __init__.py
│   ├── constants.py          # BackendName, Target enums, units
│   ├── cli.py                # typer CLI → SDK only
│   ├── sdk/
│   │   ├── runner.py         # BenchmarkRunner (public API)
│   │   ├── analytics.py          # summarize(), public exports
│   │   ├── analytics_data.py     # JSON → DataFrame + CSV
│   │   ├── analytics_plot.py     # bar charts
│   │   └── analytics_table.py    # Markdown comparison table
│   ├── services/
│   │   ├── backends/
│   │   │   ├── base.py       # InferenceBackend ABC
│   │   │   ├── ollama_backend.py
│   │   │   ├── transformers_cpu_backend.py
│   │   │   ├── airllm_backend.py
│   │   │   └── gpu_importer.py
│   │   ├── metrics/
│   │   │   ├── timer.py
│   │   │   ├── memory_monitor.py
│   │   │   └── recorder.py   # RunResult + MetricsRecorder
│   │   ├── model_provider.py
│   │   └── host_spec.py
│   └── shared/
│       ├── config.py         # pydantic-settings Settings
│       ├── gatekeeper.py     # Central external-call controller
│       └── version.py
├── tests/
│   ├── unit/                 # All mocked — fast, no real models
│   └── integration/          # Real models, opt-in via -m integration
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   └── TODO.md
├── results/                  # JSON + CSV run results
├── assets/                   # Chart PNGs
├── data/                     # Model shards (gitignored)
├── notebooks/                # Colab GPU reference notebook
├── .env-example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

---

## 🔐 Security & Secrets

- `HF_TOKEN` is read from environment / `.env` — **never** hardcoded.
- `.env` is in `.gitignore` — will never be committed.
- The Gatekeeper structured logger **never** logs token values, headers, or prompts.
- Use `.env-example` to document config; use `.env` (local, ignored) for real values.

---

## 📖 Theory: Why AirLLM Works

AirLLM implements **explicit layer-level paging of model weights**:

1. On first run, it splits the checkpoint into per-layer shard files on disk.
2. During inference, it loads **one layer at a time** into RAM, runs the forward
   pass, then discards that layer before loading the next.
3. Peak memory ≈ *one layer* (~50–150 MB) + the embedding table (~400 MB for 3B).

This is analogous to how the OS **pages process memory** to disk when physical RAM
is full — except AirLLM does it explicitly at the model-layer granularity.

The baseline (`transformers-cpu`) loads all layers simultaneously → requires
~6 GB RAM for a 3B fp16 model → OOM on a 2.5 GB free RAM host.

---

## ⚖️ Precision Caveat & Latency Context

When evaluating these benchmarks, it is critical to compare these systems honestly by accounting for differences in data precision and latency.

### The Precision Discrepancy (`fp16` vs. `q4_0`)
The benchmark involves three backends, but they do not operate on identical precision:
- **AirLLM & transformers-cpu:** Both utilize the original `fp16` (16-bit floating point) weights. This ensures maximum fidelity but inherently demands more memory and compute.
- **Ollama:** The Ollama backend serves as a sanity check but runs a highly quantized `q4_0` (4-bit integer) version of the model.

**Implication:** The Ollama sanity pass is fast and memory-efficient because the model's footprint was permanently compressed. The AirLLM result is significant because it runs the **uncompressed `fp16` model** within a constrained memory environment, achieving what `transformers-cpu` could not.

### The Latency Trade-off
Layer streaming incurs a massive I/O penalty. Every token generated requires reloading the entire model from disk, layer by layer. Our benchmark recorded a generation time of 1847.5 seconds for AirLLM (at 0.009 tokens/s).

**Conclusion:** The primary value of AirLLM is **feasibility**, not speed. It trades latency for capability, making it possible to experiment with models that would otherwise be completely inaccessible on the host hardware. The headline metric is the **reduction in peak memory**.

---

## 📜 Credits & License

- **AirLLM** by [lyogavin](https://github.com/lyogavin/airllm) — layer-by-layer inference.
- **HuggingFace Transformers** — model loading and tokenisation.
- **Ollama** — local quantised GGUF inference.
- **psutil**, **tenacity**, **pydantic-settings**, **typer**, **pandas**, **matplotlib**.

Licensed under the [MIT License](LICENSE).

---

*Generated as part of HW5 — AI Agents Orchestration course.*
