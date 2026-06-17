# AirLLM Benchmark Analysis: The Virtual Memory Framing

## Why AirLLM Fits: The Mechanism of Layer Streaming

The core premise of this benchmark is proving that large language models (LLMs) can be executed on hardware with severely constrained memory—specifically, machines where the model size vastly exceeds available physical RAM. 

When attempting to load `Qwen/Qwen2.5-3B-Instruct` (which requires approximately 6 GB in its native `fp16` format) on a host with only ~2.5 GB of free RAM, the standard `transformers-cpu` backend inevitably crashes with an Out Of Memory (OOM) error. It attempts to allocate the entire model's weight matrices into memory simultaneously before any inference can begin.

**AirLLM solves this through explicit layer-level paging of model weights.**

This mechanism operates much like an Operating System's virtual memory or pagefile, but it is optimized specifically for transformer architectures:

1. **Pre-processing / Sharding:** The model checkpoint is first split into individual shard files on disk, representing distinct transformer layers.
2. **Execution Phase (Inference):** During generation, AirLLM loads **only one layer at a time** into RAM. It computes the forward pass for that specific layer, saves the hidden states, and then explicitly discards the layer's weights from memory before loading the next one.
3. **Memory Footprint:** Peak memory consumption is drastically reduced. Instead of requiring memory for the entire model, the system only needs enough RAM to hold:
   - The embedding table (~400 MB for a 3B model)
   - A single transformer layer (~50–150 MB depending on the architecture)
   - The intermediate activations

**Result:** In our benchmarks, AirLLM successfully completed inference with a peak RSS of **2942 MB**, which is 546 MB lower than the baseline's crash point (3488 MB), thus staying safely under the physical memory ceiling.

## Precision Caveat & Latency Context

When evaluating the benchmark results, it is critical to compare these systems honestly by accounting for differences in data precision and latency.

### 1. The Precision Discrepancy (`fp16` vs. `q4_0`)

The benchmark involves three backends, but they do not operate on identical precision:
- **AirLLM & Transformers-CPU:** Both utilize the original `fp16` (16-bit floating point) weights. This ensures maximum fidelity and no quantization loss, but it inherently demands more memory and compute.
- **Ollama:** The Ollama backend serves as a sanity check but runs a highly quantized `q4_0` (4-bit integer) GGUF version of the model.

**Implication:** The Ollama sanity pass is fast and memory-efficient not because of layer streaming, but because the model's footprint was permanently compressed. The AirLLM result is significant because it runs the **uncompressed `fp16` model** within a constrained memory environment, achieving what `transformers-cpu` could not.

### 2. The Latency Trade-off

Layer streaming is not a free optimization. By treating the disk as extended memory, AirLLM incurs a massive I/O penalty:
- Every token generated requires reloading the entire model from disk, layer by layer.
- Our benchmark recorded a generation time of **1847.5 seconds** for AirLLM (at 0.009 tokens/second) compared to single-digit seconds for Ollama.

**Conclusion on Latency:** The primary value of AirLLM is **feasibility**, not speed. It trades latency for capability, making it possible to experiment with, validate, or run inference on models that would otherwise be inaccessible on the host hardware. The headline metric is the **reduction in peak memory**, proving that the model can run at all, with latency reported as the necessary cost.
