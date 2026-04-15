---
title: "KV Cache Management"
tags: [memory, kv-cache, vllm-core, offloading]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-releases.md]
related: [concepts/paged-attention.md, techniques/prefix-caching.md, techniques/fp8-quantization.md]
---

# KV Cache Management

## Summary
KV cache management encompasses all strategies for allocating, sharing, offloading, and reclaiming the key-value cache memory that stores attention state for active sequences. It is the central resource management problem in LLM serving.

## How It Works
The KV cache stores the key and value tensors for every token in every active sequence. For large models with long contexts, this dominates GPU memory usage. The management system must handle:

1. **Allocation** — [PagedAttention](paged-attention.md) allocates blocks on demand
2. **Sharing** — prefix caching and parallel sampling share blocks via reference counting
3. **Preemption** — when memory is exhausted, lower-priority sequences are evicted (swapped to CPU or recomputed)
4. **Offloading** — KV blocks can be moved to CPU memory and fetched back when needed

## Recent Developments (vLLM v0.19.0)
- **General CPU KV cache offloading** — a new mechanism with pluggable cache policy and block-level preemption handling (PRs #37160, #37874, #34805, #36642, #37853)
- **KV cache manager rethink** — Q2 2026 roadmap indicates a redesign for complex KV cache layouts
- **Disk offloading** — planned for the connector API

## Key Parameters
- `gpu_memory_utilization` — fraction of GPU memory for KV cache (default: 0.9; be cautious on shared/cloud GPUs where actual available VRAM may differ from spec)
- `swap_space` — CPU memory for swapped-out KV blocks (in GB)
- `kv_cache_dtype` — can use FP8 to halve KV cache memory

## Relationship to Other Concepts
- Built on [PagedAttention](paged-attention.md) block abstraction
- [FP8 Quantization](../techniques/fp8-quantization.md) can compress KV cache by 2x
- [Prefix Caching](../techniques/prefix-caching.md) is a sharing strategy within this system

## Open Questions
- How does the new CPU offloading compare to SGLang's approach?
- What's the right eviction policy for mixed workloads (short vs. long context)?
- How will disk offloading perform for the KV connector API?
