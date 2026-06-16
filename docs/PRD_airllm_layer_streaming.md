# PRD — AirLLM Layer Streaming Mechanism

> Per-mechanism Product Requirements Document  
> Part of: `docs/PRD.md` ecosystem for `airllm-bench`

---

## 1. What is AirLLM Layer Streaming?

AirLLM is an inference library that enables running large language models on
hardware where the total model size exceeds available RAM, by streaming
transformer layers from disk **one at a time**.

### 1.1 The Core Idea

A transformer model (e.g. Qwen2.5-3B) consists of:
- An **embedding table**: maps token IDs to vectors (~400 MB for 3B vocabulary)
- **N decoder layers**: each is a self-contained block of attention + FFN weights
- A **language model head**: projects hidden states back to vocabulary logits

Standard loading (e.g. `from_pretrained()`) loads **all** of these into RAM
simultaneously. For a 3B fp16 model: ~6 GB required.

AirLLM instead:
1. Splits the checkpoint into **per-layer shard files** on disk (first run only).
2. At inference time, loads **one layer**, runs the forward pass, **unloads**
   that layer, then loads the next.
3. Peak RAM ≈ embedding + one layer ≈ 400 MB + 150 MB ≈ ~550 MB per layer pass.

---

## 2. Memory Mathematics

For `Qwen/Qwen2.5-3B-Instruct` (fp16):

| Component | Size (fp16) |
|-----------|-------------|
| Embedding table (32K vocab × 2048 dim) | ~262 MB |
| 1 decoder layer (attention + FFN at 2048/8192) | ~50–150 MB |
| 36 decoder layers (Qwen2.5-3B) | ~5.4 GB total |
| LM head (tied to embedding) | ~262 MB |
| **Total (standard load)** | **~6 GB** |
| **AirLLM peak** | **~550 MB + embedding** |

On the target host (2.5 GB free RAM):
- Standard: **FAIL** (needs 6 GB, has 2.5 GB)
- AirLLM: **SUCCESS** (needs ~550 MB peak per layer)

---

## 3. The Sharding Process (First Run)

On the very first `AirllmBackend.load()` call:

1. `airllm.AutoModel.from_pretrained()` detects no shards at `SHARDS_PATH`.
2. Downloads the full checkpoint (or reads from HF cache).
3. Splits each transformer layer into a separate `.safetensors` file.
4. Saves shards to `data/shards/<model_name>/splitted_model/`.

On subsequent runs, the shards are reused — no re-splitting.

**First-run split time** (measured separately): ~287 seconds on the target host.

---

## 4. Inference Flow (Per Token)

```
For each generation step:
  For each layer i in [0, N-1]:
    1. load_layer(i)         ← read shard_i from disk into RAM
    2. hidden = layer_i(hidden)   ← forward pass (CPU)
    3. unload_layer(i)       ← free RAM
  Apply LM head → logits → argmax → token_id
  Decode token_id → text
```

**Consequence:** `generate_time_s` for 16 tokens involves `16 × N` layer
load/compute/unload cycles = `16 × 36 = 576` disk-load operations for a 3B model.

This explains the **extreme latency** (1847 s for 16 tokens) — it is the
expected and documented cost of staying within 2.5 GB RAM on a slow i3 CPU.

---

## 5. Trade-offs (Documented for Report)

| Dimension | AirLLM | transformers-cpu | Ollama (q4) |
|-----------|--------|-----------------|-------------|
| Peak RAM | ~2.9 GB ✅ | ~3.5 GB ❌ OOM | ~612 MB ✅ |
| Latency | Very high (1847 s / 16 tok) | N/A (OOM) | 4.2 s / 16 tok |
| Precision | fp16 | fp16 | q4 (lossy) |
| Disk usage | Model × 2 (original + shards) | Model × 1 | q4 GGUF |
| GPU support | CPU + GPU | CPU + GPU | CPU (llama.cpp) |

**Precision honesty note:** AirLLM (fp16) and transformers-cpu (fp16) are
comparable in quality. Ollama (q4) is 4-bit quantised — lower quality but much
faster and uses far less memory. The headline comparison for this assignment
focuses on **feasibility** and **peak memory**, with latency reported in context.

---

## 6. Virtual Memory Analogy

AirLLM is an **application-level explicit pager** for model weights:

| OS Concept | AirLLM Equivalent |
|------------|-------------------|
| Page | One transformer layer |
| Page file | `data/shards/` on disk |
| Page fault | `load_layer(i)` call |
| Page eviction | `unload_layer(i)` / `del layer` |
| Working set | Embedding + current layer |
| TLB hit | Layer already in RAM (not applicable — AirLLM always evicts) |

The key insight: the OS's virtual memory system would page *process* memory to
disk automatically, but at a coarse granularity and not tuned for transformer
inference patterns. AirLLM does it **explicitly and optimally** — it knows
exactly which layer is needed next and for how long.

---

## 7. ADR-1 — `compression=None` on CPU

**Decision:** Use `compression=None` (full fp16) rather than 4-bit or 8-bit
quantisation via bitsandbytes.

**Rationale:**
- `bitsandbytes` 4-bit / 8-bit quantisation requires CUDA.
- The target host has no CUDA GPU (Intel UHD integrated, no CUDA support).
- `compression=None` → fp16 layer streaming is the only viable path.

**Trade-off accepted:** Each layer is larger in fp16 than in 4-bit, so more
disk I/O per step. But it *runs*, whereas bitsandbytes would fail at import.

---

## 8. Acceptance Criteria

- AC1. `AirllmBackend.load()` calls `AutoModel.from_pretrained` with
  `device='cpu'`, `compression=None`, and `layer_shards_saving_path` set.
- AC2. First run populates `data/shards/<model>/splitted_model/` with shard files.
- AC3. `generate()` returns a non-empty string for `MAX_NEW_TOKENS=16`.
- AC4. Peak RSS measured by `MemoryMonitor` stays below `MEMORY_CEILING_MB`.
- AC5. `RunResult.status == "success"` and `output_preview` is non-null.
- AC6. Unit test: mocked `AutoModel` → correct kwargs (`device='cpu'`, `compression=None`).
