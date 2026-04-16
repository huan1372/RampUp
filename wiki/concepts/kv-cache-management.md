---
title: "KV Cache Management"
tags: [memory, kv-cache, vllm-core, offloading]
created: 2026-04-14
updated: 2026-04-16
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-releases.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-15-async-kv-prefetch-arxiv.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md]
related: [concepts/paged-attention.md, techniques/prefix-caching.md, techniques/fp8-quantization.md, techniques/kv-cache-quantization.md]
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

## Recent Developments

### vLLM v0.18.0
- **Smart CPU KV offloading** — stores only frequently-reused KV cache blocks; FlexKV as new offloading backend (source: raw/2026-04-15-vllm-v019-release.md)

### vLLM v0.19.0
- **General CPU KV cache offloading** — pluggable cache policy interface with block-level preemption handling (PRs #37160, #37874, #34805, #36642, #37853) (source: raw/2026-04-15-vllm-v019-release.md)
- **KV cache manager rethink** — Q2 2026 roadmap indicates a redesign for complex KV cache layouts
- **Disk offloading** — planned for the connector API

### Research: Async KV Cache Prefetching (arXiv 2504.06319)
A new technique orthogonal to vLLM's existing stack: schedule KV cache prefetching into GPU L2 cache during active computation windows, hiding HBM latency behind computation.
- **2.15× attention kernel efficiency** improvement
- **Up to 1.97× end-to-end throughput** on NVIDIA H20
- Surpasses FlashAttention-3 baseline
- Compatible with PagedAttention and existing kernels (source: raw/2026-04-15-async-kv-prefetch-arxiv.md)

Note: H20 results are relevant because H20 is widely deployed in Chinese cloud (export-control compliant), making this operationally significant for a large portion of production deployments.

### KV Cache Quantization: TurboQuant (merged April 15, 2026)
PR #38479 merged sub-FP8 KV cache compression into vLLM main. The technique applies WHT rotation + Lloyd-Max quantization to keys and uniform quantization to values, achieving 2.6–4.9× compression ratios.

- **`turboquant_k8v4`**: 2.6× compression; TPOT faster on long contexts (135.2 ms vs 138.1 ms @ 8K, Qwen3-4B, RTX PRO 6000 Blackwell); 79–100% throughput vs baseline
- **`turboquant_4bit_nc`**: 3.8× compression; quality risk higher; requires validation
- **`turboquant_3bit_nc`**: 4.9× compression; 0% GSM8K reported on some models — avoid without extensive testing
- **Key finding**: asymmetric K/V bit allocation (more bits for keys) consistently outperforms symmetric at the same total budget

Full details at [KV Cache Quantization](../techniques/kv-cache-quantization.md). (source: raw/2026-04-16-turboquant-kv-compression-pr38479.md)

## Key Parameters
- `gpu_memory_utilization` — fraction of GPU memory for KV cache (default: 0.9; be cautious on shared/cloud GPUs where actual available VRAM may differ from spec)
- `swap_space` — CPU memory for swapped-out KV blocks (in GB)
- `kv_cache_dtype` — can use FP8 to halve KV cache memory

## Relationship to Other Concepts
- Built on [PagedAttention](paged-attention.md) block abstraction
- [FP8 Quantization](../techniques/fp8-quantization.md) can compress KV cache by 2× with negligible overhead
- [KV Cache Quantization](../techniques/kv-cache-quantization.md) covers the full spectrum including TurboQuant (2.6–4.9×)
- [Prefix Caching](../techniques/prefix-caching.md) is a sharing strategy within this system

## Open Questions
- How does the new CPU offloading compare to SGLang's approach?
- What's the right eviction policy for mixed workloads (short vs. long context)?
- How will disk offloading perform for the KV connector API?
- Can the async KV prefetch technique (arXiv 2504.06319) be integrated into vLLM's attention kernels? Who would drive it — the SIG Performance team?
- What does the "KV cache manager rethink" in Q2 2026 actually change about the PagedAttention block abstraction?
