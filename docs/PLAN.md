# PLAN — Architecture & Engineering Design (airllm-bench)

> Companion to `docs/PRD.md`. Defines architecture, components, data schemas,
> APIs, experiment method, ADRs, testing, and security.
> Status: Draft v1.0

---

## 1. Architecture Overview

The system is a **layered benchmark harness**. All business logic lives in
**services** and is exposed through a thin **SDK**. CLI and notebooks call only
the SDK. Every external call (HuggingFace Hub, Ollama REST) is funneled through a
single **API Gatekeeper**.

### 1.1 C4 — Level 1 (Context)

```
            +---------------------+         +------------------------+
  User ───▶ |  CLI / Notebook     | ──────▶ |  airllm-bench (SDK)    |
            +---------------------+         +-----------+------------+
                                                        │
                              +-------------------------+-------------------------+
                              │                         │                         │
                       HuggingFace Hub            Ollama (local REST)       Local compute
                       (model download)           localhost:11434          (torch / airllm)
```

### 1.2 C4 — Level 2 (Containers / packages)

```
src/airllm_bench/
├── sdk/                # public API (BenchmarkRunner, facades)
├── services/
│   ├── backends/       # B1..B4 inference backends (one file each)
│   ├── metrics/        # memory monitor, timing, recorder
│   └── model_provider  # resolve/download models via gatekeeper
├── shared/
│   ├── config.py       # pydantic-settings (env/.env)
│   ├── gatekeeper.py   # central external-call control
│   └── version.py
├── constants.py        # enums, defaults, units
└── cli.py              # argparse/typer → calls SDK only
```

### 1.3 Control flow (one benchmark)

```
CLI → SDK.BenchmarkRunner.run(backend)
   → model_provider.ensure(model)        # via Gatekeeper → HF Hub
   → backend.load()                      # MemoryMonitor sampling starts
   → backend.generate(prompt, max_new)   # timing + TTFT captured
   → MetricsRecorder.write(result)       # results/*.json + *.csv
```

---

## 2. Component Responsibilities

| Component | Responsibility | Key collaborators |
|-----------|----------------|-------------------|
| `sdk.BenchmarkRunner` | Orchestrate a run/run_all; assemble `RunResult` | services, metrics |
| `services.backends.base.InferenceBackend` | ABC: `load()`, `generate()`, `name`, `teardown()` | torch/airllm/ollama |
| `OllamaBackend` (B1) | Talk to Ollama via Gatekeeper REST | gatekeeper |
| `TransformersCpuBackend` (B2) | Full-model fp16 load on CPU (baseline) | transformers |
| `AirllmBackend` (B3) | `airllm.AutoModel`, `device='cpu'`, `compression=None` | airllm |
| `GpuResultImporter` (B4) | Import Colab-produced metrics JSON | — |
| `metrics.MemoryMonitor` | Background thread sampling peak RSS / system mem | psutil |
| `metrics.Timer` | Wall-clock, load time, TTFT, generate time | — |
| `metrics.MetricsRecorder` | Persist results (JSON+CSV), host metadata | pandas |
| `model_provider` | Resolve model id → local path; pre-check disk | gatekeeper |
| `shared.config.Settings` | Typed config from env/.env | pydantic-settings |
| `shared.gatekeeper.Gatekeeper` | Retries, rate limit, queue, backpressure, logging | tenacity |
| `cli` | Parse args, call SDK, print summary | sdk |

**Rule:** GUI/CLI/notebooks never import services directly — only `sdk`.

---

## 3. Public SDK API (stable surface)

```python
class BenchmarkRunner:
    def __init__(self, settings: Settings): ...
    def run(self, backend: BackendName) -> RunResult: ...
    def run_all(self, backends: list[BackendName]) -> list[RunResult]: ...

def summarize(results: list[RunResult]) -> pandas.DataFrame: ...
def plot_comparison(df, out_path: Path) -> Path: ...
```

`BackendName` is an enum in `constants.py`:
`OLLAMA | TRANSFORMERS_CPU | AIRLLM | GPU`.

---

## 4. Data Schemas

### 4.1 `RunResult` (persisted as JSON; flattened to CSV)

```python
@dataclass
class RunResult:
    backend: str            # BackendName value
    model_id: str
    prompt_chars: int
    max_new_tokens: int
    status: str             # "success" | "failed"
    failure_reason: str | None     # e.g. "OOM", "timeout"
    load_time_s: float | None
    ttft_s: float | None           # time to first token (if available)
    generate_time_s: float | None
    total_runtime_s: float | None
    tokens_per_s: float | None
    peak_process_rss_mb: float | None
    peak_system_used_mb: float | None
    output_preview: str | None     # first N chars of generation
    host: dict              # cpu, total_ram, free_ram_at_start, os, python
    timestamp: str
    airllm_bench_version: str
```

### 4.2 Outputs
- `results/run_<backend>_<timestamp>.json` — one per run.
- `results/comparison.csv` — appended/merged table across runs.
- `assets/comparison_<metric>.png` — bar charts (runtime, peak RSS).
- `results/host_spec.json` — captured once per session.

---

## 5. API Gatekeeper Design

A single chokepoint for **all** external I/O so policy is centralized and
testable. Configured entirely from `Settings` (no hardcoded values).

Responsibilities:
- **Retries**: `tenacity` exponential backoff on transient errors (network, 5xx).
- **Rate limiting**: token-bucket / min-interval between calls, value from config.
- **Queue + backpressure**: bounded `Semaphore`; callers block when at capacity.
- **Logging/monitoring**: structured log per call (target, latency, outcome) —
  **never** logs tokens/secrets/headers.
- **Targets**: (a) HF Hub download (`huggingface_hub`), (b) Ollama REST.

```python
class Gatekeeper:
    def __init__(self, settings: Settings): ...
    def call(self, target: Target, fn: Callable, *a, **kw): ...  # wraps retry+limit
```

Inference compute (torch/airllm) is **local** and does not pass through the
Gatekeeper, but its *model download* does.

---

## 6. Configuration & Secrets

`shared/config.py` via `pydantic-settings`. Source order: env vars → `.env` →
defaults. `.env` is git-ignored; `.env-example` documents every key.

| Key | Meaning | Example |
|-----|---------|---------|
| `HF_TOKEN` | HuggingFace token (gated models) | (secret) |
| `MODEL_ID` | Primary "large" model | `meta-llama/Llama-3.2-3B-Instruct` |
| `SANITY_MODEL_ID` | Tiny model for pipeline check | `Qwen/Qwen2.5-0.5B-Instruct` |
| `OLLAMA_SANITY_MODEL` | Ollama tag for B1 sanity | `qwen2:0.5b` |
| `PROMPT` | Benchmark prompt | "Explain virtual memory in one sentence." |
| `MAX_NEW_TOKENS` | Generation length (small!) | `16` |
| `MEMORY_CEILING_MB` | Watchdog abort threshold | `3500` |
| `RUN_TIMEOUT_S` | Per-run timeout | `3600` |
| `SAMPLE_INTERVAL_S` | Memory sampling period | `0.25` |
| `OLLAMA_BASE_URL` | Ollama REST endpoint | `http://localhost:11434` |
| `HF_RATE_LIMIT_PER_MIN` | Gatekeeper rate cap | `30` |
| `SHARDS_PATH` | AirLLM layer-shard dir | `data/shards` |
| `RESULTS_DIR` / `ASSETS_DIR` | Output locations | `results/` / `assets/` |

Secrets rule: token only via env; never printed, never in `RunResult`, never in
logs. Validate inputs (model id format, positive ints, path safety).

---

## 7. Experiment Method (how numbers are produced)

Fix the controlled variables across all backends:
- Same `MODEL_ID`, same `PROMPT`, same `MAX_NEW_TOKENS`, same seed.
- One warm-up token discarded where feasible (note: AirLLM warm-up is costly).

Measured per run (definitions are identical across backends):
- **load_time_s** — model ready to generate.
- **ttft_s** — first generated token (decode start); may be N/A for Ollama.
- **generate_time_s** — full generation of `MAX_NEW_TOKENS`.
- **total_runtime_s** = load + generate (+ split time noted separately for AirLLM).
- **tokens_per_s** = `MAX_NEW_TOKENS / generate_time_s`.
- **peak_process_rss_mb / peak_system_used_mb** — sampled by `MemoryMonitor`.

Honesty note (documented in report): AirLLM runs **fp16** layer-by-layer; Ollama
runs **q4 GGUF**. Therefore the headline comparison emphasizes **feasibility**
and **peak memory**, with latency reported but contextualized by precision.

### 7.1 Expected result shape
- B2 transformers-cpu (large): `status=failed`, reason `OOM` (or extreme swap).
- B3 airllm (large): `status=success`, high `total_runtime_s`, low `peak_rss`.
- B1 ollama (large): success but slow (paging); sanity (tiny) fast.
- B4 gpu (Colab): success, low latency, high VRAM — imported JSON.

---

## 8. Architecture Decision Records (ADRs)

- **ADR-1 — AirLLM `compression=None` on host.** 4/8-bit needs bitsandbytes+CUDA,
  unavailable on Intel/Windows CPU. Decision: full fp16 layer streaming. Trade-off:
  more per-layer memory & disk, but it actually runs. *Accepted.*
- **ADR-2 — GPU baseline via Google Colab (T4).** Host has no CUDA GPU. Decision:
  produce B4 numbers on Colab with identical model/prompt/tokens; import JSON.
  Fallback: mark B4 N/A with theoretical estimate. *Accepted.*
- **ADR-3 — Primary model = 3B-class (stretch 7B/8B).** 2.5 GB free RAM + large
  7B embeddings (~1 GB) risk OOM even in AirLLM. Decision: 3B primary, 1.5B
  fallback, 7B/8B stretch. *Accepted.*
- **ADR-4 — Ollama for sanity + 2nd CPU reference.** Matches assignment stage 2;
  uses q4 GGUF (different precision) — documented. *Accepted.*
- **ADR-5 — Python 3.11 + uv.** Avoid newest Python; reproducible, locked deps.
  *Accepted.*
- **ADR-6 — Memory watchdog.** A run exceeding `MEMORY_CEILING_MB` is aborted &
  recorded as failed, protecting the host from freeze. *Accepted.*
- **ADR-7 — torch CPU build.** Install CPU-only wheels to avoid CUDA bloat and
  match the host. *Accepted.*

---

## 9. Testing Strategy

TDD where practical (Red → Green → Refactor). External deps are mocked.

| Layer | What we test | How |
|-------|--------------|-----|
| `config` | env parsing, defaults, validation errors | monkeypatched env |
| `gatekeeper` | retry on transient, rate-limit spacing, backpressure | fake clock + stub fn |
| `metrics.MemoryMonitor` | peak tracking, start/stop, sampling | injected sampler |
| `metrics.Timer/Recorder` | timing math, JSON/CSV schema | tmp_path |
| `backends.base` | interface contract | fake backend |
| `OllamaBackend` | request build, error mapping | mocked HTTP |
| `TransformersCpuBackend` | OOM → failed RunResult mapping | mocked loader raising |
| `AirllmBackend` | params (`device='cpu'`, `compression=None`), success path | mocked AutoModel |
| `BenchmarkRunner` | orchestration, run_all aggregation, failure isolation | mocked backends |
| `summarize/plot` | df shape, file written | tmp_path |

- Heavy/real-model runs are **integration**, opt-in via marker (`-m integration`),
  excluded from the coverage gate to keep CI light.
- Targets: **≥85% global coverage**; `ruff check .` zero violations.

---

## 10. Security & Safety Review

- Secrets: HF token only via env/.env; `.gitignore` covers `.env`, `*.pem`,
  `*.key`, `data/`, model caches. Never log token/headers/prompts-with-secrets.
- Input validation: model id pattern, positive integer bounds, path traversal
  checks on `SHARDS_PATH`/output dirs.
- Least privilege: read-only HF scope token; no write APIs called.
- Failure containment: timeouts + memory watchdog prevent host lockup.
- Dependency risk: pin via `uv.lock`; CPU-only torch wheels.

---

## 11. Deployment / Run Topology

- Local: `uv sync` → `uv run airllm-bench run --backend airllm` (etc.).
- Ollama: separate native install; harness talks to it over REST.
- Colab (B4 only): a notebook in `notebooks/` runs the same SDK path on T4 and
  exports `results/run_gpu_*.json`, copied back into the repo for the report.

---

## 12. Open Questions

- Q1. Does the grader require a *true* 7B "won't fit" test, or is a 3B sufficient
  given the 2.5 GB ceiling? (Assume 3B; stretch to 7B.)
- Q2. Is Colab acceptable for the GPU column, or must GPU be N/A? (Assume Colab.)
- Q3. Acceptable max wall-time for the AirLLM run during grading? (Bound via
  `RUN_TIMEOUT_S`; document actual.)

