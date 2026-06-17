# Benchmark Comparison

Cross-backend comparison of response time, peak memory, and total runtime.

| Backend | Model | Status | Response Time (s) | Peak Memory (MB) | Total Runtime (s) | Tokens/s |
|---------|-------|--------|-------------------|------------------|-------------------|----------|
| GPU | Qwen2.5-3B-Instruct | success | 2.67 | 1894.2 | 4.86 | 5.984 |
| CPU | Qwen2.5-3B-Instruct | failed (OOM) | — | 3487.9 | 47.31 | — |
| AirLLM | Qwen2.5-3B-Instruct | success | 1847.52 | 2941.7 | 2160.36 | 0.009 |
| Ollama | qwen2:0.5b | success | 4.21 | 612.4 | 7.06 | 3.797 |
