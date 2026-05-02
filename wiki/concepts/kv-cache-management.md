---
title: "KV Cache Management"
tags: [memory, kv-cache, vllm-core, offloading, cuda-graph, sequential-compression, tiering, cpu-gpu, eviction, information-theory, swa, hma]
created: 2026-04-14
updated: 2026-05-02
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-releases.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-15-async-kv-prefetch-arxiv.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-21-fp16-kv-divergence-arxiv.md, raw/2026-04-21-yoco-plus-arxiv.md, raw/2026-04-22-vllm-prs-apr21-22.md, raw/2026-04-22-sequential-kv-trie-arxiv.md, raw/2026-04-24-ttkv-arxiv.md, raw/2026-04-24-hybridgen-arxiv.md, raw/2026-04-26-vllm-prs-apr25-26.md, raw/2026-04-27-vllm-prs-apr26-27.md, raw/2026-05-01-arxiv-2604-25975-capkv.md, raw/2026-05-01-vllm-prs-may1.md, raw/2026-05-02-vllm-prs-may2.md]
related: [concepts/paged-attention.md, concepts/deepseek-v4-attention.md, techniques/prefix-caching.md, techniques/fp8-quantization.md, techniques/kv-cache-quantization.md, techniques/cross-layer-kv-compression.md, techniques/cpu-gpu-hybrid-attention.md]
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

### FP16 KV Cache: Numerical Non-Equivalence (arXiv 2604.15409, April 2026)

A key correctness finding: FP16 KV-cached inference is **not numerically equivalent** to cache-free recomputation. Cache-ON and cache-OFF execution paths traverse floating-point operations in different orders; FP16 non-associativity makes results deterministically differ.

Empirical impact across LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B on GSM8K:
- **100% token divergence rate** between cache-ON and cache-OFF paths, including under greedy decoding
- Cache-ON produces **higher accuracy in 8 of 9 conditions** — the divergence is systematically favorable, not noise
- FP32 reduces divergence by 8 orders of magnitude; FP32 flip rate = 0.0%

Architecture-specific patterns: GQA models (Mistral, LLaMA-2) show sharp divergence at layer 0; Gemma-2 (large head dim + sliding window) shows uniform drift across all layers.

**Implications**:
1. Correctness tests comparing cache-ON vs. cache-OFF in FP16 will always show 100% divergence even for bug-free implementations — not a sign of a bug
2. Any benchmark comparing cached vs. non-cached runs is confounded by this systematic divergence
3. Quantization adds additional divergence on top of this baseline; the "FP8 KV is near-zero quality impact" claim holds empirically but should be understood as marginal additional divergence beyond baseline FP16

(source: raw/2026-04-21-fp16-kv-divergence-arxiv.md)

### Cross-Layer KV Compression: YOCO++ (arXiv 2604.13556, April 2026)

An architectural approach to KV compression that eliminates per-layer KV caches for a subset of layers. YOCO++ achieves ~50% KV reduction by having top-half Transformer layers share a single global KV with learned residual connections from the bottom layer, achieving SOTA quality among cross-layer methods. Unlike quantization, requires model retraining.

Full details at [Cross-Layer KV Compression](../techniques/cross-layer-kv-compression.md). (source: raw/2026-04-21-yoco-plus-arxiv.md)

### Multi-Group KV Offload (PR #38453, April 22, 2026)

Extends the CPU-GPU KV cache offloading handler to support multi-group KV cache transfers. Previously, the offload path assumed a single KV cache group; architectures using multiple groups (e.g., MLA with separate compressed and uncompressed KV groups, or heterogeneous attention designs) could not use the offloader. This PR refactors `transfer_async` to handle group-specific sizes and block indices, with bounds checking. No performance benchmark numbers — correctness fix only.

(source: raw/2026-04-22-vllm-prs-apr21-22.md)

### HMA Multi-Group KV Offload Store (PR #39403, April 25, 2026)

Part 11 of an ongoing HMA (Hierarchical Multi-token Attention) KV offload series. Extends the offloading scheduler's `build_connector_meta` to support **storing** KV blocks across multiple groups:

- Calculates offloadable tokens per KV group independently
- Manages block indices with group-specific partitioning during storage
- Maintains bounds checking across all groups

**Limitation:** Does not support sliding window attention or SSMs — scoped to standard attention patterns.

**Context:** This enables KV cache offloading for HMA-class architectures (e.g., DeepSeek V4's CSA/HCA multi-tier KV layout). Prior to this series, the KV offloader could not function with multi-group KV cache layouts at all. No benchmark numbers — prerequisite infrastructure PR.

(source: raw/2026-04-26-vllm-prs-apr25-26.md)

### HMA KV Offload Series: Complete (PR #41445, May 1, 2026)

PR #41445 is the **final PR (13/N)** in the HMA KV offload series. The offloading connector now advertises `SupportsHMA`, formally marking the series complete. HMA is defined as models with: multiple KV cache groups, sliding window attention, or varying block sizes across groups.

**Capabilities unlocked:**
- Full-attention + SWA hybrid architectures (Mistral-class, Gemma variants) — can now use KV offloading
- MLA-based architectures with CSA+HCA+mHC multi-tier KV layout (DeepSeek V4 family)
- Any model with heterogeneous block sizes across attention groups

**Validated:** Qwen 3.6 B model with `--kv-offloading-size 16` confirmed working by external user.

**Series milestones in full:**

| PR | Date | Change |
|----|------|--------|
| #38453 | Apr 22 | Multi-group KV transfer |
| #39403 | Apr 25 | Multi-group KV store |
| #41228 | May 1 | SWA group support in scheduler |
| #41445 | May 1 | Final connector enablement (`SupportsHMA`); unit + e2e tests |

Prior to this series, KV offloading was limited to single-group, uniform-block architectures only.

(source: raw/2026-05-02-vllm-prs-may2.md)

### HMA KV Offload Scheduler: Sliding Window Group Support (PR #41228, May 1, 2026)

Part 12 of the HMA (Hybrid Multi-token Attention / Multi-group KV) offload series. Adds scheduler-side support for **sliding window attention (SWA) groups** within the KV offload path.

**Background:** The HMA series is progressively extending the KV offload scheduler to handle multi-group KV cache architectures (e.g., DeepSeek V4's CSA+HCA+mHC layout). Prior PRs established multi-group transfer (#38453, April 22) and multi-group store (#39403, April 25). However, the offload scheduler did not know how to handle SWA groups, where only the most recent W tokens are "active" and older blocks can be freed — a property the offloader must track per-group independently.

**Change:** The scheduler now tracks the sliding window horizon for each SWA group in a multi-group KV layout, correctly computing which blocks are offloadable vs. must remain in fast memory. Without this, attempting to combine KV offloading with hybrid full + SWA architectures would produce incorrect block accounting.

**Scope:** Correctness prerequisite (no benchmark numbers) for models combining full-attention layers with SWA layers when using KV offloading. Enables future offloading for architectures like Mistral-class SWA or DeepSeek hybrid designs.

**Series context:**
- Part 11 (PR #39403, Apr 25): multi-group KV offload store
- PR #40946 (Apr 27): SWA scheduler admission deadlock fix (separate fix, pre-offload path)
- Part 12 (PR #41228, May 1): SWA group support in the offload scheduler itself

(source: raw/2026-05-01-vllm-prs-may1.md)

### SWA/Chunked-Local Scheduler Admission Deadlock Fix (PR #40946, April 27, 2026)

On **hybrid full + sliding-window attention (SWA)** models (e.g., Mistral, Gemma variants) and models using chunked local attention, the runtime admission gate could reject long prompts that would actually fit in the KV pool, causing a **scheduler deadlock** — the request was perpetually refused even with sufficient memory.

**Root cause:** Two code paths computed KV block requirements with mismatched formulas:
1. **Startup pool sizer** — correctly accounted for block recycling (`SlidingWindowManager.remove_skipped_blocks` frees out-of-window blocks during chunked prefill)
2. **Runtime admission gate** — did not account for this recycling; overestimated required blocks

When the pool was sized at the startup minimum (exactly large enough for the recycling-aware worst case), the admission gate's overestimate caused deadlock.

**Fix:** A unified `max_admission_blocks_per_request` method on both `SlidingWindowSpec` and `ChunkedLocalAttentionSpec` gives one source of truth for both paths. `max_num_batched_tokens` is threaded through the KV cache coordination layers so the recycling-aware bound is available at admission time.

**Impact:** Correctness fix. No performance change. Affects SWA and chunked-local attention models with long prompts.

(source: raw/2026-04-27-vllm-prs-apr26-27.md)

### CUDAGraph Memory Profiling Default (PR #38284, April 21, 2026)

Enables CUDAGraph memory profiling by default for all vLLM serving instances. Previously opt-in via environment variable; now on by default.

**Rationale:** Large MoE models in distributed configs (specifically DeepSeek-R1 at DP=8, EP configurations) trigger OOM during startup without profiling because CUDA graph memory consumption is underestimated. Memory profiling provides accurate accounting during graph capture, preventing these startup OOMs.

**Side effect:** Default `--gpu-memory-utilization` raised from **0.9 to 0.92** to accommodate the profiling overhead. Users on memory-constrained GPUs should be aware. Logging added to warn of OOM risk if the feature is explicitly disabled.

(source: raw/2026-04-22-vllm-prs-apr21-22.md)

### Sequential KV Cache Compression (arXiv 2604.15356, April 2026)

A framework that argues per-vector KV quantization (FP8, TurboQuant) is bounded by per-vector Shannon entropy, and proposes breaking this limit by treating the KV cache as a sequence. Uses Probabilistic Language Tries to deduplicate semantically equivalent prefixes across sessions (generalizing vLLM's prefix caching), then applies predictive delta coding: each KV vector is stored as its residual from the model's own prediction. Compression ratio improves with context length.

No vLLM integration; theoretical/algorithmic paper. Full details at [KV Cache Quantization](../techniques/kv-cache-quantization.md).

(source: raw/2026-04-22-sequential-kv-trie-arxiv.md)

### Temporal-Tiered KV Cache: TTKV (arXiv 2604.19769, April 2026)

A tiering approach that maps the human memory system onto the KV cache. Recent KV states act as "short-term memory" (stored in fast HBM), while older states form "long-term memory" (stored in slower DRAM). The core observation: attention patterns in long-context tasks show that recent tokens receive the most attention weight; older tokens are accessed sparsely and selectively.

**Three-part design:**
1. **Tier Layout:** HBM (fast/high-precision) for recent tokens; DRAM (slow/lower-precision) for old tokens
2. **Tier Content:** Tier assignment based purely on temporal proximity — the most recent W tokens stay in HBM; a sliding window moves older tokens to DRAM
3. **Tier Interaction:** Block-wise streaming attention overlaps DRAM→HBM communication with HBM attention computation, hiding cross-tier latency

**Results on 128K-context tasks:**
- **5.94× cross-tier traffic reduction**
- **76% latency reduction**
- **2× throughput improvement** over strong baselines

No vLLM integration confirmed. Would require tiered block table and streaming attention kernel.

(source: raw/2026-04-24-ttkv-arxiv.md)

### CPU-GPU Hybrid Attention: HybridGen (arXiv 2604.18529, April 20, 2026)

Extends the offloading idea further: the CPU is not just a storage medium but an **active compute participant**. HybridGen splits attention computation across CPU and GPU simultaneously on systems with CXL-expanded memory.

**Mechanism:** GPU computes attention over HBM-resident KV (recent/important); CPU computes attention over CXL-DRAM-resident KV (older/less-important). Results merged via log-sum-exp aggregation identical to FlashAttention's tile reduction. A feedback-driven scheduler adjusts the split boundary dynamically.

**Semantic-Aware Placement:** KV blocks are assigned to HBM or CXL-DRAM based on estimated importance from prior decode-step attention scores.

**Results:**
- **1.41×–3.2× throughput improvement** over 6 KV cache management baselines
- Full context fidelity (no eviction) — unlike pruning/eviction methods
- Evaluated on 3 models × 11 sizes × 3 GPU platforms with CXL memory

See full details at [CPU-GPU Hybrid Attention](../techniques/cpu-gpu-hybrid-attention.md). No vLLM integration confirmed.

(source: raw/2026-04-24-hybridgen-arxiv.md)

### Depth–Cache Tradeoffs for Reasoning (arXiv 2604.17935, April 20, 2026)

Formal analysis of how aggressively the KV cache can be compressed before multi-step reasoning degrades. Uses k-hop pointer chasing on n tokens as the reasoning proxy task, under a shared KV cache of size s and a locality-respecting cache controller. Key results:

- **Upper bound (constructive):** `L = O(min(k, ⌈k/s⌉ log s) · log n/(mp))` — achievable via windowed pointer doubling
- **Lower bound (conjectured):** `L = Ω(⌈k/s⌉ · ⌈log₂ n/(Hmp)⌉)` — remaining gap is a probabilistic step on cache trace joint distribution
- **Max-bound (unconditional):** `L = Ω(max(⌈k/s⌉, log n/(Hmp)))`

Practical implication: for k-step reasoning tasks, compressing KV cache to s < √n/4 requires model depth to scale as Ω(k/s). Aggressive KV compression below this threshold may force the model to implicitly re-encode intermediate reasoning state in residual activations, requiring deeper models.

(source: searched arXiv 2604.17935; full raw source not yet ingested due to limited fetchable content)

### KV Cache Eviction: CapKV (arXiv 2604.25975, April 28, 2026)

CapKV rethinks KV cache eviction — the decision of which token-level KV entries to drop when cache capacity is exceeded — through the **Information Bottleneck (IB)** principle, replacing ad-hoc heuristics (attention score accumulators used by H2O, SnapKV, ScissorHands) with a theoretically grounded objective.

**Formulation:** Under a linear-Gaussian surrogate of attention, the optimal retained KV subset S maximizes mutual information I(Y; KV_S) between future attention output Y and retained KV pairs. This reduces to:

> maximize log det(I + σ⁻² Kₛᵀ Kₛ) over subset S

where Kₛ is the key matrix of retained entries — a **log-determinant maximization** problem.

**CapKV implementation:** Approximates the log-det objective using **statistical leverage scores** — diagonal elements of the hat matrix H = K(KᵀK)⁻¹Kᵀ. Leverage score for entry i measures how much that entry influences the attention output: high leverage = hard to reconstruct from remaining entries. O(n) per eviction step.

**Key insight vs prior heuristics:** Attention-score accumulators (H2O, SnapKV) approximate historical query relevance — a proxy for "was this entry attended to in the past?" CapKV approximates information capacity — "if this entry is dropped, how much does the output distribution shift?" These diverge at long contexts, explaining why heuristic methods degrade at 32K+ token sequences.

**Results:** Consistently outperforms H2O, SnapKV, ScissorHands, Ada-KV across multiple models and long-context benchmarks; largest gains at high compression ratios. Specific accuracy numbers and model names not captured (arXiv HTML unavailable; sourced from search summaries).

**vLLM integration status:** None. Requires modifications to the KV block allocator to run leverage score computation per step.

**Relationship to other eviction/compression work:**
- **GRACE (arXiv 2604.16983):** channel-level pruning (removes KV dimensions); CapKV is token-level eviction (removes entire token entries). Different axes — can be composed.
- **TurboQuant/KV quantization:** compresses retained entries; orthogonal to eviction.
- **TTKV (arXiv 2604.19769):** tiering without eviction; CapKV evicts. Complementary strategies.

(source: raw/2026-05-01-arxiv-2604-25975-capkv.md)

## Key Parameters
- `gpu_memory_utilization` — fraction of GPU memory for KV cache (**default: 0.92** as of PR #38284, April 21, 2026; previously 0.9; be cautious on memory-constrained GPUs)
- `swap_space` — CPU memory for swapped-out KV blocks (in GB)
- `kv_cache_dtype` — can use FP8 to halve KV cache memory

## Relationship to Other Concepts
- Built on [PagedAttention](paged-attention.md) block abstraction
- [FP8 Quantization](../techniques/fp8-quantization.md) can compress KV cache by 2× with negligible overhead
- [KV Cache Quantization](../techniques/kv-cache-quantization.md) covers the full spectrum including TurboQuant (2.6–4.9×)
- [Prefix Caching](../techniques/prefix-caching.md) is a sharing strategy within this system
- [Cross-Layer KV Compression](../techniques/cross-layer-kv-compression.md) — architectural approach reducing KV count across layers (YOCO++)
- [CPU-GPU Hybrid Attention](../techniques/cpu-gpu-hybrid-attention.md) — HybridGen: CPU as active compute participant for KV-resident attention (arXiv 2604.18529)

## Open Questions
- Now that the HMA KV offload series (PR #41445) is complete, what is the throughput overhead of KV offloading for DeepSeek V4-Pro (multi-tier CSA+HCA layout) vs. single-group architectures?
- Does `--kv-offloading-size` need to be tuned per architecture class (single-group vs. SWA vs. multi-group MHC) for optimal offloading efficiency?
- How does CapKV's leverage-score approximation scale with context length? At what n does O(n) per step become a bottleneck relative to attention compute?
- Can CapKV be combined with TTKV (temporal tiering) — use tiering to partition "keep in HBM / move to DRAM" and CapKV to decide what to evict entirely?
- Does the IB formulation in CapKV degrade for multi-query or grouped-query attention where the key matrix structure differs from the linear-Gaussian surrogate assumption?
- How does TTKV's temporal tiering (arXiv 2604.19769) compare to HybridGen's semantic-aware placement? Can they be combined?
- What is TTKV's sliding window size W (tokens in HBM) and how is it tuned for a given workload?
- Does HybridGen's CPU attention work for reasoning tasks where far-back context is retroactively important?
- When will CXL memory be commodity-level in cloud (prerequisite for HybridGen deployment)?
- How does the new CPU offloading compare to SGLang's approach?
- What's the right eviction policy for mixed workloads (short vs. long context)?
- How will disk offloading perform for the KV connector API?
- Can the async KV prefetch technique (arXiv 2504.06319) be integrated into vLLM's attention kernels? Who would drive it — the SIG Performance team?
- What does the "KV cache manager rethink" in Q2 2026 actually change about the PagedAttention block abstraction?
- Does KV block offloading to CPU compound the FP16 divergence (arXiv 2604.15409) via additional memory copy operations?
- When will vLLM support YOCO-family architectures (cross-layer KV sharing)? Does this require the Q2 2026 KV cache manager rethink?
- Does the depth–cache tradeoff (arXiv 2604.17935) hold empirically for current reasoning models (DeepSeek-R1, o3)? At what compression ratio does reasoning actually degrade?
- Can vLLM's prefix caching be extended to approximate/semantic prefix matching (as in arXiv 2604.15356 Layer 1)?
- What is the performance impact of CUDAGraph memory profiling default (PR #38284) on startup time vs. runtime efficiency?
