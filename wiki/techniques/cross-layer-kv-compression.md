---
title: "Cross-Layer KV Compression"
tags: [kv-cache, compression, architecture, attention, research]
created: 2026-04-21
updated: 2026-04-21
sources: [raw/2026-04-21-yoco-plus-arxiv.md]
related: [concepts/kv-cache-management.md, techniques/kv-cache-quantization.md, techniques/fp8-quantization.md]
---

# Cross-Layer KV Compression

## Summary

Cross-layer KV compression reduces KV cache memory by sharing or approximating KV tensors across Transformer layers instead of storing independent KVs per layer. Unlike quantization-based approaches ([FP8](fp8-quantization.md), [TurboQuant](kv-cache-quantization.md)), cross-layer methods are **architectural** — they require training the model with a modified attention structure. The main representative is the YOCO family, with YOCO++ (arXiv 2604.13556, April 2026) achieving state-of-the-art quality at 50% compression. See also [KV Cache Management](../concepts/kv-cache-management.md) for the broader memory management context.

## Problem It Solves

Standard Transformers store one KV cache per layer. For a 32-layer model with 128 heads, head dimension 128, and FP16 dtype, KV cache per token = 32 × 2 × 128 × 128 × 2 bytes ≈ 2 MB/token. At 8K sequence length and 64 concurrent sequences, this is ~1 TB — exceeding a single node.

Quantization (FP8, TurboQuant) compresses per-layer KVs in-place. Cross-layer methods instead eliminate some layers' KV caches entirely by sharing a common KV from another layer.

## How It Works

### YOCO (You Only Cache Once)

The base YOCO method divides Transformer layers into two halves:

- **Bottom half (local layers)**: standard per-layer KV computation and caching
- **Top half (global layers)**: all layers share the KV tensors from a single designated "global" attention layer (typically the middle layer)

Result: only one KV tensor is stored for the entire top half → ~50% total KV cache reduction.

Limitation: quality degrades because top-half layers lose per-layer KV specificity. The middle layer's KV is a lossy summary.

### YOCO++ (arXiv 2604.13556)

YOCO++ adds a **weighted residual connection** between each bottom-half layer's KV and the bottom-layer (input-level) KV:

```
KV_layer_i = f(input_i) + alpha_i * KV_layer_0
```

where `alpha_i` is a learned scalar weight per layer. The combined KV is **cached** directly (not recomputed on the fly), preserving the same inference memory access pattern as YOCO while improving capacity.

Key engineering detail: caching the combined KV avoids additional I/O overhead during decode — the inference path reads one blended tensor, same as YOCO.

**Performance (April 2026 results)**:
- 50% KV cache compression (same as baseline YOCO)
- SOTA quality among cross-layer methods at the 50% operating point
- Outperforms plain YOCO on downstream tasks at identical training/inference efficiency
- Claims to outperform a standard Transformer at the same parameter count (architectural inductive bias benefit)

(source: raw/2026-04-21-yoco-plus-arxiv.md)

## Implementation in vLLM

**Current status (April 2026)**: NOT supported in vLLM. Cross-layer KV attention requires a non-standard attention pattern that vLLM's V1/MRV2 backends do not implement for arbitrary shared-KV layouts.

For a YOCO++ model to run on vLLM, the engine would need to:
1. Recognize the YOCO++ attention pattern from model config
2. Allocate KV cache blocks only for bottom-half layers (eliminating top-half per-layer blocks)
3. Route top-half attention to use the shared global KV block

The Q2 2026 vLLM roadmap item "KV cache manager rethink" could enable this, but no confirmed work has started.

## Comparison with Other KV Compression Methods

| Method | Requires retraining | Compression | Quality risk | Production-ready (vLLM) |
|--------|---------------------|-------------|--------------|-------------------------|
| FP8 KV cache | No | 2× | Minimal | Yes |
| TurboQuant k8v4 | No | 2.6× | Low | Main branch only |
| TurboQuant 3-bit | No | 4.9× | High | Experimental |
| YOCO | Yes (architecture) | ~2× | Moderate | No |
| YOCO++ | Yes (architecture) | ~2× | Low (SOTA) | No |
| xKV (SVD-based) | No (post-hoc) | varies | Varies | No |

Cross-layer methods trade deployment flexibility (requires model retrain) for avoiding online quantization overhead.

## Trade-offs

- **Gain**: 50% KV cache reduction with no inference-time compute overhead on the decode path
- **Lose**: requires training a new model from scratch or fine-tuning with the YOCO++ architecture; not applicable to existing deployed models
- **Gain**: better quality than YOCO at same compression; competitive with full-KV Transformer
- **Lose**: not yet supported in production serving systems (vLLM, SGLang, TRT-LLM)

## When to Use

- When training a new model specifically for memory-constrained deployment
- When quantization-based compression (FP8, TurboQuant) is insufficient or introduces quality risk
- Not applicable to existing model weights without retraining

## Open Questions

- When will vLLM's "KV cache manager rethink" enable YOCO-family models?
- What are the exact perplexity/benchmark numbers for YOCO++ vs. full-KV Transformer?
- Does YOCO++ interact favorably with FP8 quantization on top of the 50% cross-layer compression?
- Can YOCO++ be applied as a fine-tuning step to an existing Transformer, or does the residual connection require full pretraining?

## Sources

- [raw/2026-04-21-yoco-plus-arxiv.md](../../raw/2026-04-21-yoco-plus-arxiv.md) — primary source for YOCO++ technique
