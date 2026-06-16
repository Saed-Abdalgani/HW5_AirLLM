# PRD — HW5: Proving AirLLM Value on a Constrained Machine

> Product Requirements Document
> Project codename: **airllm-bench**
> Status: Draft v1.0 · Owner: Student · Course: AI Agents Orchestration — HW5 (AirLLM)

---

## 1. Purpose

Empirically **prove that AirLLM enables running a model that otherwise does not
fit in available memory**, on a low-resource, GPU-less laptop, and quantify the
trade-off: it *runs*, but at the cost of latency.

We deliver a small, reproducible **benchmark harness** that runs the *same model*
through multiple execution backends and produces a comparison of **response
time**, **memory consumption**, and **total runtime** across **GPU**, **CPU
(baseline)**, and **AirLLM (CPU, layer-by-layer)**.

The deliverable is both:
1. A working, tested, documented Python project (per the engineering standard).
2. A results report (tables + charts + narrative) that answers the assignment.

---

## 2. Target Hardware (the constraint that defines the project)

| Resource | Value | Implication |
|----------|-------|-------------|
| CPU | Intel Core i3-1005G1 @ 1.20 GHz, 2 cores / 4 threads | Very slow inference; keep token counts tiny |
| RAM (total) | ~8 GB | OS + apps consume ~5–6 GB |
| RAM (free for HW) | ~2.5 GB (measured ~1.8 GB free) | Hard memory ceiling; primary design constraint |
| GPU | Intel UHD Graphics (integrated, ~1 GB shared) | **No CUDA** → no local GPU baseline |
| Disk free | ~82 GB on C: | Enough for layer shards of a 3B–8B model |
| OS | Windows | bitsandbytes 4bit needs CUDA → unavailable |
| Python | 3.11.9 | Stable, broad package compatibility (not newest) |

**Consequences (verified, see PLAN ADRs):**
- AirLLM must run with `device='cpu'` and `compression=None` (4/8-bit needs CUDA).
- The GPU baseline must come from **Google Colab (free T4)** or be documented N/A.
- Model size is bounded by **peak single-layer + embedding** memory, not total
  model size; large-vocab 7B embeddings (~1 GB) are risky at 2.5 GB free.

---

## 3. Users & Stakeholders

| User | Need |
|------|------|
| Course grader | Clear evidence AirLLM works where baseline fails; reproducible run; report |
| Student (author) | Understand AirLLM internals, virtual-memory analogy, measurement method |
| Future reuse | A clean, modular harness to benchmark new backends/models |

---

## 4. Goals & Non-Goals

### 4.1 Goals
- G1. Verify the local pipeline with a tiny model via **Ollama** (sanity check).
- G2. Establish a **baseline** where a "reasonably large" model **fails or runs
  very slowly** because it does not fit in ~2.5 GB free RAM.
- G3. Run the **same model with AirLLM on CPU** and show it **completes**.
- G4. **Measure & compare** response time, memory, runtime across GPU / CPU /
  AirLLM and present tables + charts + analysis.
- G5. Ship as an engineered project: SDK + Gatekeeper, config/secrets safety,
  tests ≥85% coverage, ruff clean, full docs, `uv` tooling.

### 4.2 Non-Goals
- No model training / fine-tuning (LoRA/QLoRA are background context only).
- No production serving, no web UI, no multi-user concurrency.
- No claim that AirLLM is *fast*; the point is *feasibility under memory limits*.
- No attempt to make bitsandbytes 4bit work on CPU/Windows.

---

## 5. Success Metrics / KPIs

| KPI | Definition | Target |
|-----|-----------|--------|
| K1 Feasibility | AirLLM completes generation of the large model on the host | PASS (sequence returned) |
| K2 Baseline contrast | transformers-CPU baseline OOMs or is ≥10× slower / infeasible | Documented failure or extreme slowdown |
| K3 Peak memory (AirLLM) | Peak process RSS during AirLLM run | < free RAM + pagefile; ideally ≤ ~3–4 GB |
| K4 Reproducibility | Fresh `uv sync` + documented steps reproduce results | Reproducible |
| K5 Quality gate | Tests ≥85% coverage, ruff zero violations | PASS |
| K6 Report | Comparison table + ≥1 chart + written analysis | Delivered in `results/` + `assets/` |

---

## 6. Scope — Execution Backends (the comparison matrix)

Same `model`, same `prompt`, same tiny `max_new_tokens` (default 16, configurable).

| ID | Backend | Device | Precision / format | Loads all layers at once? | Expected outcome |
|----|---------|--------|--------------------|---------------------------|------------------|
| B1 | `ollama` | CPU | q4 GGUF (llama.cpp) | mmap + paging | Sanity (tiny) ✓; large → very slow |
| B2 | `transformers-cpu` | CPU | fp16, all in RAM | **Yes** | **Baseline FAIL (OOM)** for large model |
| B3 | `airllm` | CPU | fp16, layer-by-layer | **No** (1 layer at a time) | **Runs**, high latency ✓ |
| B4 | `gpu` (Colab) | CUDA T4 | fp16 | Yes | Fast reference (off-host) |

> B1 (Ollama) doubles as the assignment's "install Ollama + basic run" stage and
> as a second CPU reference. B4 is off-host because the laptop has no CUDA GPU.

---

## 7. Functional Requirements

- FR1. **Config-driven**: model id, prompt, `max_new_tokens`, paths, rate limits,
  sampling interval come from config/env — never hardcoded.
- FR2. **Secret handling**: HuggingFace token read from environment / `.env`
  (git-ignored); never logged or committed; `.env-example` provided.
- FR3. **Backend abstraction**: a common interface `InferenceBackend` with
  `load()` and `generate()`, implemented by B1–B4.
- FR4. **Metrics capture** per run: model-load time, time-to-first-token (where
  available), generate time, total runtime, tokens/sec, peak process RSS, peak
  system memory used, success/failure + error class.
- FR5. **Gatekeeper**: all external calls (HF Hub download, Ollama REST API) go
  through one component handling retries, rate limits (from config), queueing,
  backpressure, and logging.
- FR6. **SDK entry point**: `BenchmarkRunner` exposes `run(backend)` and
  `run_all()`; CLI and notebooks call only the SDK.
- FR7. **Result persistence**: structured results to `results/` (JSON + CSV);
  charts to `assets/`; raw logs retained.
- FR8. **Graceful failure**: an OOM/timeout in one backend is captured as a
  recorded result (status=failed, reason), not a crash of the whole run.
- FR9. **Reproducibility**: fixed seed, pinned deps (`uv.lock`), documented host
  spec captured into the results metadata.

---

## 8. Non-Functional Requirements

- NFR1. **Memory safety**: harness must monitor and bound memory; the AirLLM run
  must stay within the host's free RAM + pagefile. A configurable watchdog aborts
  a run that exceeds a memory ceiling instead of freezing the OS.
- NFR2. **Modularity**: each source file ≤150 actual code lines; one
  responsibility per module; business logic only in services/SDK.
- NFR3. **Testability**: every module/public function tested; external deps
  (HF Hub, Ollama, AirLLM, torch) mocked in unit tests; ≥85% global coverage.
- NFR4. **Lint/style**: `ruff check .` zero violations; docstrings on public API.
- NFR5. **Tooling**: `uv` only (`uv sync`, `uv add`, `uv run …`); no pip/venv.
- NFR6. **Observability**: structured logging with levels; per-run log file;
  no secrets in logs.
- NFR7. **Portability of measurement**: same metric definitions across backends
  so numbers are comparable (document any unavoidable apples-vs-oranges gaps,
  e.g. AirLLM fp16 vs Ollama q4).
- NFR8. **Time budget**: a single benchmark run with default tiny token count
  should finish within a documented, bounded wall time; long runs are expected
  for AirLLM and must be timeout-guarded, not unbounded.

---

## 9. Assumptions

- A1. Network access is available to download models/shards from HuggingFace.
- A2. A free Google Colab account is available for the GPU reference run (B4).
  If not, B4 is reported as **N/A (no CUDA device on host)** with rationale.
- A3. Ollama can be installed on the host (Windows) for B1.
- A4. The grader accepts a 3B-class model as "reasonably large" given the
  2.5 GB ceiling; 7B/8B is attempted as a stretch goal only.
- A5. Disk has room for full fp16 weights + per-layer shards of the chosen model.

---

## 10. Constraints

- C1. **No CUDA on host** → no local GPU run; AirLLM `compression=None`.
- C2. **~2.5 GB free RAM** is the hard ceiling for B2/B3 on the host.
- C3. **Windows** → avoid CUDA-only bitsandbytes paths; watch long-path limits
  for many per-layer shard files.
- C4. **Slow CPU** → `max_new_tokens` kept very small (default 16) for all runs.
- C5. **Python 3.11** pinned for package compatibility.
- C6. Secrets must never be committed; `.env` is git-ignored.

---

## 11. Dependencies (high level; exact versions pinned in uv.lock)

- Core: `airllm`, `transformers`, `torch` (CPU build), `accelerate`,
  `safetensors`, `huggingface_hub`.
- Tooling/runtime: `psutil` (memory), `tenacity` (retries), `pydantic-settings`
  (config), `typer` or `argparse` (CLI), `pandas`+`matplotlib` (report).
- Quality: `pytest`, `pytest-cov`, `ruff`.
- External tools: **Ollama** (separate installer), **Google Colab** (B4 only).

---

## 12. Milestones (phased — see TODO for task-level detail)

| Phase | Milestone | Exit criteria |
|-------|-----------|---------------|
| P0 | Project scaffolding + tooling | `uv sync` works; structure + config + `.env-example` in place |
| P1 | Sanity pipeline | Tiny model runs via Ollama **and** transformers-cpu; metrics captured |
| P2 | Baseline failure | Large model via transformers-cpu → OOM/very-slow, recorded as result |
| P3 | AirLLM success | Same large model completes via AirLLM CPU; metrics captured |
| P4 | GPU reference | Same model+prompt run on Colab T4; metrics exported & imported |
| P5 | Measure & compare | Results table + chart(s) + written analysis in `results/`/`assets/` |
| P6 | Hardening & audit | Tests ≥85%, ruff clean, README + per-mechanism PRD, final checklist |

---

## 13. Acceptance Criteria

- AC1. Running the documented commands reproduces: a tiny-model success, a
  baseline large-model failure/slowdown, and an AirLLM large-model success.
- AC2. A single comparison artifact (CSV + chart) shows response time, peak
  memory, and runtime for GPU / CPU / AirLLM (and Ollama) on the same model.
- AC3. Written analysis explains *why* AirLLM fits where the baseline does not,
  using the virtual-memory / layer-streaming concept.
- AC4. No secrets in repo; `.env-example` present; `.env` git-ignored.
- AC5. `uv run pytest` passes with ≥85% coverage; `uv run ruff check .` clean.
- AC6. README enables a new user to install and reproduce end-to-end.

---

## 14. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | AirLLM OOM even layer-by-layer (big embedding) | Med | High | Fall back 3B→1.5B; close apps; raise pagefile |
| R2 | bitsandbytes 4bit fails on CPU/Windows | High | Low | Use `compression=None` by design (documented) |
| R3 | No Colab access for GPU numbers | Low | Med | Report B4 as N/A with theoretical estimate |
| R4 | AirLLM run takes hours on 2-core CPU | Med | Med | tiny `max_new_tokens`; timeout watchdog; document latency |
| R5 | Disk fills during layer splitting | Low | Med | Pre-check free space; `delete_original`; pick smaller model |
| R6 | Ollama install/permissions on Windows | Low | Low | Document install; fallback to transformers-cpu sanity |
| R7 | Apples-vs-oranges (fp16 vs q4) skews compare | Med | Low | Document precision per backend; compare feasibility+memory primarily |

---

## 15. Glossary

- **AirLLM**: inference lib that streams transformer layers from disk one at a
  time, so peak memory ≈ one layer instead of the whole model.
- **Layer sharding**: splitting a checkpoint into per-layer files on first run.
- **Prefill / Decode**: prompt-encoding pass vs token-by-token generation.
- **Peak RSS**: maximum resident memory of the process during a run.
- **Virtual memory analogy**: AirLLM is an explicit, smart form of paging —
  it pages *model weights* in/out of RAM the way the OS pages process memory.

