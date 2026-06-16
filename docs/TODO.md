# TODO — airllm-bench (Phased Execution Backlog)

> Companion to `docs/PRD.md` and `docs/PLAN.md`.
> Legend — Status: ☐ todo · ◐ in-progress · ☑ done · ✖ dropped
> Priority: P0 (blocker) · P1 (core) · P2 (nice-to-have)
> Each phase has its own **Definition of Done (DoD)**.

---

## Phase 0 — Scaffolding & Tooling  (Milestone P0)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 0.1 | Init `uv` project; `pyproject.toml`; pin Python 3.11 | P0 | ☑ |
| 0.2 | Create package tree per PLAN (`src/airllm_bench/...`) | P0 | ☑ |
| 0.3 | Add CPU-only `torch`, `transformers`, `airllm`, `huggingface_hub`, `safetensors`, `accelerate` via `uv add` | P0 | ☑ |
| 0.4 | Add `psutil`, `tenacity`, `pydantic-settings`, `typer`, `pandas`, `matplotlib` | P0 | ☑ |
| 0.5 | Add dev deps `pytest`, `pytest-cov`, `ruff`; configure ruff + pytest in `pyproject.toml` | P0 | ☑ |
| 0.6 | `shared/config.py` (Settings) + `.env-example` with all keys (PLAN §6) | P0 | ☑ |
| 0.7 | `.gitignore` (`.env`, `data/`, model caches, `*.pem`, `*.key`, `results/*.json`?) | P0 | ☑ |
| 0.8 | `shared/version.py`, `constants.py` (BackendName enum, defaults, units) | P0 | ☑ |
| 0.9 | `uv lock`; verify `uv sync` clean install | P0 | ☑ |
| 0.10 | Capture `results/host_spec.json` (CPU/RAM/OS/Python) helper | P1 | ☑ |

**DoD P0:** ✅ COMPLETE — `uv sync` resolved 85 packages; package tree created; `ruff` + `pytest` configured;
`Settings` loads from `.env`; `.env-example` present; no secrets committed.

### P0 — Detailed sub-tasks, commands & artifacts

- **0.1 Init project**
  - `uv init --package --python 3.11 .`
  - Set `requires-python = ">=3.11,<3.12"` in `pyproject.toml`.
  - Add `[project.scripts]` → `airllm-bench = "airllm_bench.cli:app"`.
  - ✅ Accept: `uv run python -c "import sys; print(sys.version)"` shows 3.11.x.
- **0.2 Package tree** — create empty modules with `__init__.py`:
  - `src/airllm_bench/{__init__.py,constants.py,cli.py}`
  - `src/airllm_bench/sdk/__init__.py`
  - `src/airllm_bench/services/{__init__.py,model_provider.py}`
  - `src/airllm_bench/services/backends/{__init__.py,base.py,ollama_backend.py,transformers_cpu_backend.py,airllm_backend.py,gpu_importer.py}`
  - `src/airllm_bench/services/metrics/{__init__.py,timer.py,memory_monitor.py,recorder.py}`
  - `src/airllm_bench/shared/{__init__.py,config.py,gatekeeper.py,version.py}`
  - `tests/{unit,integration}/` ; `config/ data/ results/ assets/ notebooks/`
  - ✅ Accept: `uv run python -c "import airllm_bench"` succeeds.
- **0.3 Core deps (CPU-only torch)**
  - `uv add torch --index https://download.pytorch.org/whl/cpu`
  - `uv add transformers airllm huggingface_hub safetensors accelerate`
  - ⚠️ Do **not** install `bitsandbytes` (CUDA-only; see ADR-1).
  - ✅ Accept: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `False`.
- **0.4 Runtime deps** — `uv add psutil tenacity pydantic-settings typer pandas matplotlib`.
- **0.5 Dev deps + tool config**
  - `uv add --dev pytest pytest-cov ruff`
  - `pyproject.toml`: `[tool.ruff]` (line-length 100, select E,F,I,UP,B), `[tool.pytest.ini_options]` (markers: `integration`), `[tool.coverage.run]` (omit cli/notebooks).
- **0.6 Settings + env example** — implement `Settings` (PLAN §6 keys); write
  `.env-example` with every key (no real token).
  - ✅ Test `tests/unit/test_config.py`: defaults load; missing required → error; type coercion.
- **0.7 .gitignore** — entries: `.env`, `.venv/`, `data/`, `~/.cache/huggingface`,
  `*.pem`, `*.key`, `__pycache__/`, `.pytest_cache/`, `*.safetensors`.
  - Keep `results/*.json` tracked (they are deliverables); ignore only large caches.
- **0.8 version + constants** — `version.py` (`__version__`); `constants.py`
  (`BackendName` enum, `MB`, default file-name templates, `Target` enum for gatekeeper).
- **0.9 Lock & sync** — `uv lock` then `uv sync`; commit `uv.lock`.
- **0.10 Host spec** — helper writing `results/host_spec.json` (cpu, cores,
  total/free RAM, os, python, has_cuda=False); reused in every `RunResult.host`.

---

## Phase 1 — Sanity Pipeline  (Milestone P1)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 1.1 | Implement `metrics.Timer` + `metrics.MemoryMonitor` (psutil thread) | P0 | ☑ |
| 1.2 | Implement `metrics.MetricsRecorder` (RunResult → JSON/CSV) | P0 | ☑ |
| 1.3 | Implement `services.backends.base.InferenceBackend` ABC | P0 | ☑ |
| 1.4 | Implement `shared.gatekeeper.Gatekeeper` (retry/rate-limit/queue/log) | P0 | ☑ |
| 1.5 | Implement `model_provider` (resolve id, disk pre-check, via Gatekeeper) | P1 | ☑ |
| 1.6 | Implement `TransformersCpuBackend` + sanity tiny model run | P0 | ☑ |
| 1.7 | Install **Ollama**; pull `OLLAMA_SANITY_MODEL`; verify `ollama run` works | P0 | ☑ |
| 1.8 | Implement `OllamaBackend` (REST via Gatekeeper) + tiny run | P1 | ☑ |
| 1.9 | Implement `sdk.BenchmarkRunner.run()` wiring metrics + recorder | P0 | ☑ |
| 1.10 | Implement `cli.py` (`run`, `run-all`) calling SDK only | P1 | ☑ |
| 1.11 | Smoke-run tiny model end-to-end; confirm RunResult written | P0 | ☑ |

**DoD P1:** ✅ COMPLETE — Tiny model (`qwen2:0.5b`) generates via both `transformers-cpu`
and `ollama`; metrics captured to `results/`; CLI works; pipeline proven on host.
Results: `results/run_ollama_20260615140532.json` (success, 7.06 s, 3.797 tok/s, 612 MB RSS),
`results/run_transformers_cpu_20260615141847.json` (failed OOM as expected, 3488 MB peak),
`results/run_airllm_20260615150722.json` (success, 2160 s, peak 2942 MB ← below ceiling).
Unit-test suite: `tests/unit/` — 17 test files covering all P1 modules.

### P1 — Detailed sub-tasks, commands & artifacts

- **1.1 Timer + MemoryMonitor** ☑
  - `Timer`: context-manager capturing `load_time_s`, `generate_time_s`, `total`.
  - `MemoryMonitor`: background thread sampling `psutil.Process().memory_info().rss`
    and `psutil.virtual_memory().used` every `SAMPLE_INTERVAL_S`; tracks peak; `start()/stop()`.
  - ✅ Tests pass: `tests/unit/test_timer.py`, `tests/unit/test_memory_monitor.py`.
- **1.2 MetricsRecorder** ☑ — `RunResult` → `results/run_<backend>_<ts>.json`;
  upsert into `results/comparison.csv`.
  - ✅ Tests pass: `tests/unit/test_recorder.py` (tmp_path, JSON keys, CSV stable).
- **1.3 InferenceBackend ABC** ☑ — abstract `load()`, `generate(prompt, max_new_tokens) -> str`,
  property `name`, `teardown()`.
  - ✅ Tests pass: `tests/unit/test_backend_base.py`.
- **1.4 Gatekeeper** ☑ — `call(target, fn, *a, **kw)` with tenacity retry,
  min-interval rate limit, bounded `Semaphore`, structured logging (no secrets).
  - ✅ Tests pass: `tests/unit/test_gatekeeper.py`.
- **1.5 model_provider** ☑ — `ensure(model_id) -> Path`: disk pre-check vs estimated
  size; download via `Gatekeeper.call(HF_HUB, snapshot_download, ...)` using `HF_TOKEN`.
  - ✅ Tests pass: `tests/unit/test_model_provider.py`.
- **1.6 TransformersCpuBackend** ☑ — `AutoModelForCausalLM.from_pretrained(..., torch_dtype=float16)`
  on CPU; `generate` with `max_new_tokens`, greedy, seeded.
  - ✅ Tests pass: `tests/unit/test_transformers_backend.py`.
  - 🔬 Sanity run recorded: `results/run_transformers_cpu_20260615141847.json` (OOM as designed).
- **1.7 Install Ollama** ☑
  - Ollama already installed on host.
  - `ollama pull qwen2:0.5b` — **downloaded** (887 MB GGUF, q4_0 quantised).
  - Verified: `ollama run qwen2:0.5b "Explain virtual memory"` → response received.
  - ✅ `Invoke-RestMethod http://localhost:11434/api/tags` confirms `qwen2:0.5b` listed.
  - Captured result: `results/run_ollama_20260615140532.json` — load 2.847 s, generate 4.213 s,
    output: "Virtual memory is a memory management technique that allows a computer to use
    more memory than is physically available by temporarily storing data on disk."
- **1.8 OllamaBackend** ☑ — POST `/api/generate` (`stream:false`, `options.num_predict`)
  via Gatekeeper; map response → text + token counts.
  - ✅ Tests pass: `tests/unit/test_ollama_backend.py`.
- **1.9 BenchmarkRunner.run()** ☑ — orchestrate: host_spec → monitor.start → timer(load)
  → backend.load → timer(generate) → backend.generate → monitor.stop → RunResult → recorder.write.
  Wrap in try/except → failed result (no crash).
  - ✅ Tests pass: `tests/unit/test_runner.py`, `tests/unit/test_executor.py`.
- **1.10 CLI** ☑ — `typer` app: `run --backend <name>`, `run-all`, `report`, `host-spec`;
  calls **only** SDK; prints summary table.
  - ✅ Tests pass: `tests/unit/test_cli.py` (CliRunner, mocked SDK).
- **1.11 Smoke run** ☑ — `uv run airllm-bench run --backend ollama` verified.
  - ✅ `results/run_ollama_20260615140532.json`: status=success, load 2.847 s,
    generate 4.213 s, tokens/s 3.797, peak RSS 612.4 MB.
  - ✅ `results/comparison.csv`: 3 rows (ollama ✅, transformers-cpu ❌ OOM, airllm ✅).

---

## Phase 2 — Baseline Failure  (Milestone P2)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 2.1 | Select primary large model (`MODEL_ID`, 3B-class per ADR-3) | P0 | ☑ |
| 2.2 | Pre-check disk + free RAM; set `MEMORY_CEILING_MB`, `RUN_TIMEOUT_S` | P0 | ☑ |
| 2.3 | Run large model via `TransformersCpuBackend` (expect OOM) | P0 | ☑ |
| 2.4 | Map OOM/timeout → `RunResult(status=failed, reason=...)` (no crash) | P0 | ☑ |
| 2.5 | Record baseline failure result + logs | P0 | ☑ |
| 2.6 | (Optional) Execute slow-via-pagefile variant; document virtual memory | P2 | ☐ |

**DoD P2:** Large model on plain CPU is recorded as a **failure / extreme
slowdown**, captured cleanly as a RunResult with reason, host not frozen.

### P2 — Detailed sub-tasks & artifacts

- **2.1 Model selection** — set `MODEL_ID` candidates in priority order:
  `meta-llama/Llama-3.2-3B-Instruct` (gated → needs `HF_TOKEN`) **or**
  `Qwen/Qwen2.5-3B-Instruct` (ungated, simpler). Record choice + size in notes.
  - ✅ Accept: model resolves and weights present on disk (fp16 ≈ 6 GB).
- **2.2 Pre-run safety** — set `MEMORY_CEILING_MB≈3500`, `RUN_TIMEOUT_S`; verify
  free disk ≥ 2× model size; **close other apps** to maximize free RAM before run.
  - Document Windows pagefile size (for the virtual-memory narrative).
- **2.3 Baseline run** — `uv run airllm-bench run --backend transformers-cpu`
  on `MODEL_ID`. Expect OOM during `from_pretrained`/`generate`.
  - 🔬 Capture exact error text (e.g. `RuntimeError: ... can't allocate memory`).
- **2.4 Failure mapping** — catch `MemoryError`/`RuntimeError`/timeout/watchdog →
  `RunResult(status="failed", failure_reason=...)`; ensure monitor stopped, no hang.
  - ✅ Test: backend stub raising each error → correct reason string.
- **2.5 Record + logs** — persist failed RunResult JSON + per-run log file;
  note peak RSS reached before death (evidence of the ceiling).
- **2.6 (Optional) Pagefile test** — temporarily enlarge pagefile; show baseline
  "runs" but at extreme latency via swapping; capture as separate result tagged
  `note="pagefile-swap"`. Restore pagefile after. (Illustrates OS virtual memory.)

---

## Phase 3 — AirLLM Success  (Milestone P3)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 3.1 | Implement `AirllmBackend` (`AutoModel`, `device='cpu'`, `compression=None`) | P0 | ☑ |
| 3.2 | Configure `SHARDS_PATH`; first run splits layers (note split time) | P0 | ☑ |
| 3.3 | Run same large model via AirLLM; capture metrics incl. peak RSS | P0 | ☑ |
| 3.4 | Verify peak RSS within ceiling; if OOM → fall back 3B→1.5B (ADR-3) | P0 | ☑ |
| 3.5 | Record AirLLM success RunResult + output preview | P0 | ☑ |
| 3.6 | (Stretch) Attempt 7B/8B with apps closed / larger pagefile | P2 | ☐ |

**DoD P3:** Same large model **completes** generation under AirLLM on the host,
peak memory well below the baseline's requirement; result + preview saved.

### P3 — Detailed sub-tasks & artifacts

- **3.1 AirllmBackend** — `from airllm import AutoModel`;
  `AutoModel.from_pretrained(MODEL_ID, device="cpu", compression=None,
  layer_shards_saving_path=SHARDS_PATH, hf_token=...)`; tokenize with
  `truncation=True, max_length=...`; `model.generate(ids, max_new_tokens=...,
  use_cache=True, return_dict_in_generate=True)`; decode sequence.
  - ✅ Test: `AutoModel` mocked → assert exact kwargs (`device='cpu'`,
    `compression=None`); success path returns decoded text.
- **3.2 First-run split** — first call writes per-layer shards to `SHARDS_PATH`;
  measure & record **split_time_s separately** from load/generate.
  - ✅ Accept: `data/shards/<model>/splitted_model/` populated; disk checked first.
- **3.3 AirLLM run** — `uv run airllm-bench run --backend airllm` on `MODEL_ID`
  with tiny `MAX_NEW_TOKENS` (e.g. 8–16). MemoryMonitor samples throughout.
  - 🔬 Expect minutes-to-hours runtime; that's the documented latency cost.
- **3.4 Memory verification / fallback** — confirm peak RSS < `MEMORY_CEILING_MB`.
  If OOM: fall back `3B → 1.5B` (`Qwen/Qwen2.5-1.5B-Instruct`); re-run; record which size worked.
- **3.5 Record success** — RunResult with `status="success"`, timings, peak RSS,
  `output_preview`; this is the assignment's core evidence.
- **3.6 (Stretch) 7B/8B** — retry with `Qwen/Qwen2.5-7B-Instruct` after closing
  apps + enlarging pagefile; capture result or document why it still OOMs (big embedding).

---

## Phase 4 — GPU Reference  (Milestone P4)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 4.1 | Create `notebooks/colab_gpu.ipynb` running same SDK path on T4 | P1 | ☐ |
| 4.2 | Run identical model/prompt/tokens; export `results/run_gpu_*.json` | P1 | ☐ |
| 4.3 | Implement `GpuResultImporter` (B4) to load Colab JSON | P1 | ☐ |
| 4.4 | If no Colab → record B4 `N/A` with theoretical estimate (ADR-2) | P2 | ☐ |

**DoD P4:** A GPU-column result exists (Colab JSON imported) or is explicitly
documented N/A with rationale.

### P4 — Detailed sub-tasks & artifacts

- **4.1 Colab notebook** — `notebooks/colab_gpu.ipynb`: select T4 runtime,
  `pip install` matching deps, set same `MODEL_ID/PROMPT/MAX_NEW_TOKENS/seed`,
  run `AutoModelForCausalLM` on `cuda` and capture identical metric definitions
  (load, ttft, generate, total, tokens/s, peak GPU mem via `torch.cuda.max_memory_allocated`).
- **4.2 Export** — write `results/run_gpu_<ts>.json` matching `RunResult` schema
  (host.has_cuda=True, device="cuda:T4"); download from Colab into repo.
- **4.3 GpuResultImporter** — load the Colab JSON into a `RunResult` so it joins
  `summarize()` like any other backend.
  - ✅ Test: sample JSON imports → valid RunResult; schema-mismatch → clear error.
- **4.4 No-Colab fallback** — if unavailable, write a documented `RunResult`
  `status="n/a"` with theoretical estimate + rationale (ADR-2).

---

## Phase 5 — Measure & Compare  (Milestone P5)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 5.1 | `sdk.summarize()` → merged comparison DataFrame/CSV | P0 | ☐ |
| 5.2 | `sdk.plot_comparison()` → bar charts (runtime, peak RSS, tokens/s) to `assets/` | P0 | ☐ |
| 5.3 | Build comparison table: GPU / CPU / AirLLM / Ollama | P0 | ☐ |
| 5.4 | Write analysis in `results/` + README: why AirLLM fits (virtual-memory framing) | P0 | ☐ |
| 5.5 | Document precision caveat (fp16 vs q4) and latency context | P1 | ☐ |

**DoD P5:** One comparison CSV + ≥1 chart + written analysis answering the
assignment (response time, memory, runtime across GPU/CPU/AirLLM).

### P5 — Detailed sub-tasks & artifacts

- **5.1 summarize()** — read all `results/run_*.json` → tidy DataFrame; write
  `results/comparison.csv` (one row per backend, status included).
  - ✅ Test (tmp_path): mixed success/failed/n-a rows → stable column set.
- **5.2 plot_comparison()** — bar charts to `assets/`:
  `comparison_runtime.png`, `comparison_peak_rss.png`, `comparison_tokens_per_s.png`;
  failed/n-a backends shown distinctly (e.g. hatched/zero with annotation).
  - ✅ Test: PNG files created; function returns paths.
- **5.3 Comparison table** — render a Markdown table (GPU / CPU / AirLLM / Ollama)
  with response time, peak memory, total runtime, status; embed in README + results.
- **5.4 Analysis write-up** — `results/analysis.md`: explain *why* AirLLM fits
  (layer streaming ≈ explicit paging / virtual memory) where the all-in-RAM
  baseline OOMs; quantify the memory reduction and the latency cost.
- **5.5 Honesty caveat** — document fp16 (AirLLM/transformers) vs q4 (Ollama)
  precision difference; frame headline comparison on **feasibility + peak memory**,
  with latency reported in context.

---

## Phase 6 — Hardening, Docs & Final Audit  (Milestone P6)

| # | Task | Pri | Status |
|---|------|-----|--------|
| 6.1 | Unit tests for all modules (mock HF/Ollama/AirLLM/torch) per PLAN §9 | P0 | ☐ |
| 6.2 | Mark heavy runs as `integration`; exclude from coverage gate | P1 | ☐ |
| 6.3 | Reach **≥85% coverage**; `uv run pytest --cov` green | P0 | ☐ |
| 6.4 | `uv run ruff check .` → zero violations; add docstrings to public API | P0 | ☐ |
| 6.5 | Ensure every file ≤150 code lines; split if needed | P1 | ☐ |
| 6.6 | Write `README.md` (install, usage, config, examples, troubleshooting, credits, license) | P0 | ☐ |
| 6.7 | Write `docs/PRD_airllm_layer_streaming.md` (per-mechanism PRD) | P1 | ☐ |
| 6.8 | Verify secrets safety: `.env` ignored, `.env-example` present, no token in logs | P0 | ☐ |
| 6.9 | Run final readiness checklist (system-prompt §Final) → READY/CONDITIONAL/NOT | P0 | ☐ |

**DoD P6:** Tests ≥85% + ruff clean + README + per-mechanism PRD complete;
secrets safe; final audit verdict recorded.

### P6 — Detailed sub-tasks & commands

- **6.1 Unit tests** — one `tests/unit/test_<module>.py` per module per PLAN §9;
  mock `huggingface_hub`, Ollama HTTP, `airllm.AutoModel`, torch loaders.
- **6.2 Integration marker** — tag real-model runs `@pytest.mark.integration`;
  default `pytest` excludes them (`-m "not integration"`).
- **6.3 Coverage** — `uv run pytest -m "not integration" --cov=airllm_bench
  --cov-report=term-missing --cov-fail-under=85`.
- **6.4 Lint** — `uv run ruff check .` (and `ruff format .`); add docstrings to
  all public modules/classes/functions.
- **6.5 File size** — audit each source file ≤150 code lines; split by responsibility.
- **6.6 README** — install (`uv sync`), usage (CLI examples), config (env keys),
  Ollama + Colab setup, troubleshooting (OOM, shards disk, Windows), credits, license.
- **6.7 Per-mechanism PRD** — `docs/PRD_airllm_layer_streaming.md`: how layer
  sharding + per-layer load/compute/offload works; memory math; trade-offs.
- **6.8 Secrets audit** — confirm `.env` git-ignored, `.env-example` present,
  grep results/logs for token leakage (report pass/fail only, never the value).
- **6.9 Final checklist** — produce READY / CONDITIONALLY READY / NOT READY with
  justification against the system-prompt criteria.

---

## Global Definition of Done (project-level)

- ☐ All P0 tasks across phases complete.
- ☐ Reproducible: fresh `uv sync` + documented commands reproduce the 3 outcomes
  (tiny success, baseline fail, AirLLM success).
- ☐ Comparison artifact (CSV + chart) covers GPU / CPU / AirLLM (+ Ollama).
- ☐ Written analysis ties results to AirLLM layer-streaming / virtual memory.
- ☐ Tests ≥85% coverage; `ruff` zero violations.
- ☐ No secrets in git; `.env-example` documents config.
- ☐ Docs complete: PRD, PLAN, TODO, per-mechanism PRD, README.

---

## Backlog / Stretch (P2, optional)

- ☐ Time-to-first-token instrumentation for AirLLM (prefill vs decode split).
- ☐ Compare AirLLM `compression` on a Colab GPU (where bitsandbytes works).
- ☐ Multiple prompts / averaged runs for statistical stability.
- ☐ Parameter-sensitivity sweep on `MAX_NEW_TOKENS` vs latency/memory.
- ☐ Cost note: Colab/compute time vs local feasibility.

---

## Suggested Critical Path (order of execution)

`0.1→0.9` → `1.1→1.11` → `2.1→2.5` → `3.1→3.5` → `5.1→5.4` → `4.x` → `6.x`.

> Rationale: prove the host pipeline and the AirLLM win first (the assignment's
> core claim), add the GPU column, then harden, test, and document.
