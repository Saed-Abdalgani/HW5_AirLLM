# PRD: AirLLM Layer Streaming Mechanism

## 1. Overview
This document specifies the AirLLM layer streaming algorithm, which is the core mechanism enabling large language models (LLMs) to execute on severely memory-constrained hosts. By explicitly paging model weights to and from disk at the granularity of a single transformer layer, this subsystem prevents the Out-Of-Memory (OOM) failures inherent to traditional monolithic loading paradigms.

## 2. Goals & Assumptions
**Goals:**
- Enable inference for a 3B+ parameter model (requiring ~6 GB in `fp16`) on a host with only 2.5 GB of free RAM.
- Maintain maximum fidelity by avoiding aggressive lossy quantization (i.e., operating on original `fp16` weights rather than `q4`).
- Transparently manage disk I/O without requiring changes to the user's inference code.

**Assumptions & Constraints:**
- The host has a fast SSD to mitigate the massive I/O penalty.
- Latency is acceptable. Speed is sacrificed for the pure capability of execution (feasibility).

## 3. Mechanism Description
The layer streaming mechanism effectively implements a targeted virtual memory manager for transformer layers.

### Phase 1: Pre-processing (Sharding)
When the model is invoked for the first time, the monolithic checkpoint is intercepted.
1. The checkpoint is parsed to isolate the embedding table, individual transformer layers (e.g., layers 0 through $N-1$), and the final output heads.
2. Each separated block is serialized to disk as an independent "shard" file within `SHARDS_PATH`.

### Phase 2: Per-Layer Load / Compute / Offload
During the forward pass of generation, the system abandons the `transformers` default loop and instead performs the following sequence for every generated token:
1. Load the embedding table into RAM.
2. Compute token embeddings and save intermediate activations.
3. For $i = 0$ to $N-1$:
   - Load shard $i$ (Layer $i$) from disk into RAM.
   - Perform the forward pass on the current activations.
   - Overwrite the intermediate activations in memory with the new output.
   - Explicitly delete Layer $i$ from RAM (triggering garbage collection).
4. Load the LM Head, compute logits, and sample the next token.

## 4. Memory Math
For `Qwen/Qwen2.5-3B-Instruct` (approx 36 transformer layers):
- **Monolithic Load (Baseline):** 
  $6 \text{ GB (Weights)} + 400 \text{ MB (KV Cache / Activations)} \approx 6.4 \text{ GB}$ (Result: OOM)
- **AirLLM Streaming:** 
  $400 \text{ MB (Embeddings)} + 150 \text{ MB (One Layer)} + 400 \text{ MB (Activations)} \approx 950 \text{ MB}$ 
  *(System overhead accounts for the measured ~2.9 GB peak RSS, but it remains well below the 3.5 GB ceiling.)*

## 5. Trade-offs & Risks
| Trade-off | Impact | Mitigation |
| :--- | :--- | :--- |
| **Latency** | Generating a single token requires reading the entire 6 GB model from disk sequentially. Total runtime is increased by a factor of ~500x over a native RAM execution. | Ensure the use case is asynchronous and latency-insensitive. |
| **SSD Wear** | Continuous reads and potential swapping generate high disk activity. | Keep generation sequences short (`MAX_NEW_TOKENS`). |
| **Windows Limits** | Shard creation leads to deep directory structures. | Document the need to enable Long Paths in the Windows Registry. |
