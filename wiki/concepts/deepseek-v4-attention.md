---
title: "DeepSeek V4 Hybrid Attention (CSA + HCA + mHC)"
tags: [attention, long-context, sparse-attention, kv-cache, deepseek, moe, architecture, tool-calling]
created: 2026-04-24
updated: 2026-04-27
sources: [raw/2026-04-24-deepseek-v4-vllm.md, raw/2026-04-26-vllm-prs-apr25-26.md, raw/2026-04-27-vllm-prs-apr26-27.md]
related: [concepts/kv-cache-management.md, concepts/paged-attention.md, techniques/kv-cache-quantization.md, techniques/disaggregated-serving.md, techniques/speculative-decoding.md]
---

# DeepSeek V4 Hybrid Attention (CSA + HCA + mHC)

## Summary

DeepSeek V4 (released April 24, 2026) introduces a hybrid attention architecture that achieves 1M-token context at 27% of V3.2's per-token FLOPs and 10% of V3.2's KV cache. The architecture combines three components: Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), and Manifold-Constrained Hyper-Connections (mHC). CSA and HCA are different levels of hierarchical KV compression with learned retrieval; mHC is a structured residual connection that provides numerical stability at extreme depth. vLLM has day-0 support (April 24, 2026).

## Models

| Model | Total Params | Active Params | KV at 1M ctx (vs V3.2) | FLOPs at 1M ctx (vs V3.2) |
|-------|-------------|---------------|------------------------|---------------------------|
| DeepSeek-V4-Pro | 1.6T | 49B | **10%** | **27%** |
| DeepSeek-V4-Flash | 284B | 13B | **7%** | **10%** |

Both are MoE architectures with 1M token context, Apache 2.0 license. (source: raw/2026-04-24-deepseek-v4-vllm.md)

## How It Works

### Compressed Sparse Attention (CSA)

CSA replaces full KV attention on old context with a two-level access pattern:

1. **Compression**: every m consecutive tokens are compressed into a single KV entry (specific m not disclosed)
2. **Retrieval**: a learned **Lightning Indexer** scores compressed blocks and selects top-k most relevant
3. **Hybrid attention**: selected compressed blocks + a sliding window of recent uncompressed tokens → full attention

Conceptually, CSA is a hierarchical memory system:
- **L1 (sliding window)**: recent uncompressed tokens — fast, exact
- **L2 (retrieved compressed blocks)**: top-k relevant past context — learned retrieval
- **Archive**: all other compressed tokens — ignored for this step

This is structurally distinct from prior sparse attention methods (e.g., Longformer local+global, BigBird random+local+global) in that the selection is content-adaptive via a learned Lightning Indexer rather than fixed-pattern.

### Heavily Compressed Attention (HCA)

HCA uses a much larger fixed compression group:
- Group size m' = **128** tokens compressed → 1 entry
- No learned retrieval — all compressed entries are attended to (there are far fewer of them at 128× compression)
- Applied to a subset of layers where coarse context representation suffices

HCA is cheaper than CSA but loses retrieval precision. The hybrid stack assigns CSA to layers that need fine-grained recall (e.g., factual lookup) and HCA to layers that need only global context summaries.

### Manifold-Constrained Hyper-Connections (mHC)

Standard residual: `y = x + F(x)`

mHC: `y = x + g(A · F(x))` where:
- `A` is a learned mixing matrix constrained to the **Birkhoff polytope** (set of doubly stochastic matrices: non-negative entries, rows and columns sum to 1)
- Birkhoff constraint → spectral norm of the residual map is bounded by ≤ 1 (non-expansive)
- Non-expansive residual → signal propagation stays numerically stable across very deep stacks

Constraint is enforced via Sinkhorn normalization during training (iterative row/column normalization converges to a doubly stochastic matrix).

**Why this matters at 1M context:** CSA/HCA models are effectively deeper in computation per token than standard attention (added indexing + retrieval steps). Activation explosions would accumulate faster. mHC's spectral norm bound prevents this.

(source: raw/2026-04-24-deepseek-v4-vllm.md)

## Implementation in vLLM

### Hybrid KV Cache Layout
Three KV categories require separate storage:
1. **Compressed CSA blocks**: stored with Lightning Indexer scoring metadata; retrieved per-query
2. **Compressed HCA blocks**: fixed-ratio compressed; all retained
3. **Recent tokens (sliding window)**: standard PagedAttention block allocation

vLLM manages all three in a unified KV cache layout. The standard PagedAttention block allocator handles the sliding window; the compressed blocks require additional metadata storage.

### Kernel Fusion
The CSA retrieval step (Lightning Indexer scoring → top-k selection → concat with sliding window) is performance-critical. vLLM fuses these operations into a single kernel to avoid materializing large intermediate attention tensors.

### Disaggregated Serving
V4-Pro's 1.6T parameters and 1M context make multi-node deployment essential. vLLM uses the KV Connector API for cross-node KV transfer. The hybrid KV cache (compressed + uncompressed) increases transfer complexity vs. standard dense-attention models.

### Relation to MLA
DeepSeek V4 retains MLA (Multi-head Latent Attention) from V3.2 for non-CSA/HCA layers. The model contains three attention types: standard MLA, CSA, and HCA. vLLM handles all three in a single model execution path.

(source: raw/2026-04-24-deepseek-v4-vllm.md)

## Key Parameters

- `m` — CSA compression group size (tokens per compressed entry); specific V4 value undisclosed
- `m'` — HCA compression group size; = **128** in V4
- `k` — top-k compressed blocks retrieved in CSA (value undisclosed)
- `w` — sliding window size (recent uncompressed tokens in CSA); value undisclosed

## Relationship to Other Concepts

- **KV Cache Management** — CSA+HCA represent a new compression paradigm at the model architecture level, distinct from post-hoc quantization (TurboQuant, FP8). See [KV Cache Management](kv-cache-management.md).
- **PagedAttention** — Sliding window portion of CSA uses standard PagedAttention block allocation. See [PagedAttention](paged-attention.md).
- **KV Cache Quantization** — CSA/HCA reduce KV size via structural compression; quantization (FP8, TurboQuant) further reduces per-entry bit width. These stack multiplicatively. See [KV Cache Quantization](../techniques/kv-cache-quantization.md).
- **Disaggregated Serving** — 1M context + 1.6T params mandates disaggregated P/D for V4-Pro. See [Disaggregated Serving](../techniques/disaggregated-serving.md).

## Benchmarks & Numbers

| Metric | V4-Pro vs V3.2 at 1M context | V4-Flash vs V3.2 at 1M context |
|--------|-------------------------------|--------------------------------|
| KV cache | 10% (−90%) | 7% (−93%) |
| FLOPs/token | 27% (−73%) | 10% (−90%) |

No external quality benchmarks (MMLU, HumanEval, etc.) available from sources collected as of 2026-04-24.

(source: raw/2026-04-24-deepseek-v4-vllm.md)

## Open Questions

- What is CSA's compression group size m and top-k retrieval count k?
- How does the Lightning Indexer work — learned dense scorer, content-hash, or attention-based?
- At what context length does V4-Pro outperform V3.2 in throughput (breakeven for CSA/HCA overhead vs KV savings)?
- What quality cost does HCA's 128-token compression impose on long-document recall tasks?
- Does mHC's Sinkhorn normalization add meaningful training overhead?
- What is the KV transfer volume in disaggregated serving for a V4-Pro 1M-token request?
- Will vLLM eventually support CSA-style retrieval for other models, or is this V4-specific?
- Does vLLM's PagedAttention block allocator handle compressed CSA blocks, or is a separate allocator used?

## Post-Release Fixes (April 26, 2026)

### DSML Token Leakage in Streaming Tool Calls (PR #40806)

In streaming tool-call scenarios, the DSML (DeepSeek Markup Language) sentinel token `｜DSML｜` was leaking into streamed content when the `<｜DSML｜tool_calls>` marker spanned response chunk boundaries. The streaming parser was emitting partial marker text before confirming the sentinel was complete.

**Fix:** Added `_extract_content()` with partial tag overlap detection and a `_sent_content_idx` cursor per request (same pattern used in KimiK2 and Glm4Moe parsers). Any text suffix that could form a prefix of the start marker is buffered until confirmed.

**Impact:** Correctness fix — eliminates garbled output in streaming tool-call responses. No throughput or latency change. Affects DeepSeek V4, V4-Flash, and V3.2 models on all GPU configurations.

(source: raw/2026-04-26-vllm-prs-apr25-26.md)

## Post-Release Fixes (April 27, 2026)

### SiLU Clamp Limit for Shared Expert (PR #40950)

DeepSeek V4's shared expert uses a SiLU (`silu_and_mul`) activation. At extreme hidden-state magnitudes, unclamped SiLU values caused numerical instability. PR #40950 introduces a new `DeepseekV4MLP` class and bakes clamp limits into the `silu_and_mul` CUDA kernel (via a `clamp_limit` parameter) rather than applying a separate tensor clamp pass.

**Impact:** Correctness fix for numerical overflow/underflow in the shared expert. Minor throughput improvement from eliminating a separate clamp kernel. Affects DeepSeek-V4, V4-Flash, and models using `DeepseekV4MLP`.

(source: raw/2026-04-27-vllm-prs-apr26-27.md)

## Sources

- [raw/2026-04-24-deepseek-v4-vllm.md](../../raw/2026-04-24-deepseek-v4-vllm.md) — primary source for all DeepSeek V4 architecture details and vLLM implementation
- [raw/2026-04-26-vllm-prs-apr25-26.md](../../raw/2026-04-26-vllm-prs-apr25-26.md) — DSML streaming fix (PR #40806)
- [raw/2026-04-27-vllm-prs-apr26-27.md](../../raw/2026-04-27-vllm-prs-apr26-27.md) — SiLU clamp for shared expert (PR #40950)
