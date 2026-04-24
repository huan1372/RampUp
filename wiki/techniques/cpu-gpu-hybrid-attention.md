---
title: "CPU-GPU Hybrid Attention"
tags: [kv-cache, cpu-gpu, hybrid, cxl, long-context, offloading, memory, attention]
created: 2026-04-24
updated: 2026-04-24
sources: [raw/2026-04-24-hybridgen-arxiv.md]
related: [concepts/kv-cache-management.md, concepts/paged-attention.md, techniques/disaggregated-serving.md, techniques/tensor-parallelism.md]
---

# CPU-GPU Hybrid Attention

## Summary

CPU-GPU hybrid attention splits the attention computation itself across CPU and GPU simultaneously, rather than simply offloading KV blocks to CPU as passive storage. The CPU actively computes attention logits over its portion of the KV cache while the GPU does the same for its portion; results are aggregated. This exploits both memory tiers (GPU HBM for recent/important KV, CPU-accessible DRAM/CXL for older/less-important KV) and both compute units.

## Problem It Solves

Million-token contexts produce KV caches of hundreds of gigabytes — far beyond single-GPU HBM capacity. Standard approaches either:
1. **KV pruning/eviction**: drop tokens permanently, losing information
2. **Sequential offloading**: move KV blocks to CPU and retrieve them serially — GPU idles during the fetch

Neither approach uses CPU compute. Hybrid attention uses CPU cycles (which are memory-bandwidth-bound and often idle during GPU attention) to compute over CPU-resident KV in parallel with GPU attention.

## How It Works

### HybridGen (arXiv 2604.18529, April 2026)

HybridGen is the primary demonstrated implementation of this technique, targeting systems with **CXL-expanded memory** (CPU-accessible DRAM pool reachable via CXL interface).

**Step 1 — Attention Logit Parallelism**: Partition the full KV cache. Recent / high-importance KV blocks stay in GPU HBM. Older / lower-importance blocks are in CXL DRAM accessible to CPU. Both devices compute attention logits over their respective KV partitions independently using the same query vector.

**Step 2 — Aggregation**: GPU collects CPU partial results via log-sum-exp reduction (identical to the reduction used in FlashAttention across tiles) to produce the final attention output.

**Step 3 — Feedback-Driven Scheduler**: A runtime scheduler monitors CPU and GPU utilization and adjusts the partition boundary dynamically. If GPU is bottlenecked, fewer KV blocks are assigned to it; if CPU is bottlenecked, the split shifts more to GPU.

**Step 4 — Semantic-Aware KV Mapping**: KV blocks are placed in GPU HBM or CXL DRAM based on estimated **semantic importance** to the current query. Importance is estimated from attention scores in prior decode steps. High-importance blocks remain in fast GPU HBM; low-importance blocks are mapped to CXL DRAM.

## Implementation in vLLM

Not yet integrated. Would require:
1. Extending PagedAttention's block table to track HBM vs CXL-DRAM blocks
2. A dual-device attention kernel (GPU-side FlashAttention + CPU-side attention with aggregation)
3. A scheduler aware of CPU memory bandwidth and compute latency

## Benchmarks

### HybridGen (Evaluated on 3 models × 11 sizes × 3 GPU platforms with CXL memory)

| Metric | Result |
|--------|--------|
| Throughput vs 6 KV cache management baselines | **1.41×–3.2× improvement** |
| Accuracy vs baselines | Superior (no KV block eviction) |

Baselines compared: six state-of-the-art KV cache management methods (exact names not available; likely include H2O, SnapKV, StreamingLLM-class approaches).

(source: raw/2026-04-24-hybridgen-arxiv.md)

## Relationship to Other Techniques

| Technique | CPU Role | KV Completeness |
|-----------|----------|----------------|
| Standard KV offloading (vLLM v0.18.0+) | Storage only; GPU fetches serially | Full (but serial) |
| CPU-GPU Hybrid Attention | Active compute participant | Full (parallel) |
| KV eviction / pruning | N/A | Partial (information loss) |
| TTKV tiering (arXiv 2604.19769) | Storage (DRAM tier) | Full (async streaming) |

See also [KV Cache Management](../concepts/kv-cache-management.md) for the full offloading/tiering landscape.

## Trade-offs

**Gain**: Full context fidelity (no KV eviction), higher throughput than serial offloading, uses otherwise-idle CPU cycles.

**Lose**: Requires CXL-memory-capable hardware (not universally available); adds CPU→GPU aggregation communication; scheduler adds latency; semantic KV importance estimation requires bookkeeping.

## When to Use

- Long-context inference (>32K tokens) where KV cache exceeds GPU HBM
- Systems with CXL-expanded memory pools (datacenter class)
- Workloads where KV eviction quality loss is unacceptable

## Open Questions

- Does semantic-aware KV mapping work for reasoning tasks where later context may become retroactively important (e.g., chain-of-thought where the conclusion requires far-back context)?
- What is the CPU→GPU aggregation latency at different partition sizes?
- Can HybridGen's aggregation be fused with PagedAttention's existing block-parallel attention?
- How does HybridGen compare to TTKV (arXiv 2604.19769) on 128K+ context tasks?
- When will CXL memory be commodity-level in cloud instances (required for HybridGen deployment)?
