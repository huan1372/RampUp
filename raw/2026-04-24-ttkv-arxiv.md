---
title: "TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.19769
collected: 2026-04-24
tags: [kv-cache, memory, tiering, long-context, throughput, latency, offloading]
---

# TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference

**arXiv:** 2604.19769  
**Submitted:** April 2026  
**Authors:** Researchers from Harbin Institute of Technology and Guangzhou University

## Abstract / Core Claim

KV cache memory scales linearly with context length, creating a bottleneck for long-context inference. TTKV maps the human memory system onto the KV cache: recent KV states act as short-term memory (critical for generation), while older states form long-term memory (only a small subset relevant to any given query). TTKV partitions the KV cache into temporal tiers with heterogeneous capacity and precision, placing recent tokens in fast/high-precision memory and older tokens in slow/lower-precision memory.

## Design Components

### Tier Layout
Decouples fast memory (HBM — on-device GPU high-bandwidth memory) from slow memory (DRAM — CPU or off-chip memory). HBM holds recent KV state; DRAM holds temporally older KV state.

### Tier Content
Assigns KV states to tiers based on **temporal proximity** — how recently each token was generated. More recent KV states go to faster, higher-precision tiers. Older tokens are moved to the slow tier as a sliding window advances.

### Tier Interaction
Block-wise streaming attention overlaps cross-tier communication with computation. When the attention kernel requires KV blocks from the slow tier, it prefetches them while computing attention on the fast-tier (recent) blocks in parallel. This hides the memory access latency of the slow tier.

## Performance Results

- **5.94× cross-tier traffic reduction** on 128K-context tasks compared to baselines
- **76% latency reduction** on 128K-context tasks
- **2× throughput improvement** over strong baselines

Evaluated on long-context tasks (128K token contexts).

## Key Insight

The observation that temporal recency predicts attention relevance is empirically robust for most long-context tasks: the model attends most heavily to the most recent tokens and a small "important" subset of old tokens. TTKV's tier structure exploits this without requiring per-token importance scoring (which would add overhead).

## Relationship to vLLM

TTKV is a research paper; no vLLM integration has been confirmed. The technique is conceptually compatible with PagedAttention's block abstraction (tier assignment is a block-level decision). Implementation would require:
1. A tiered block table (HBM blocks vs DRAM blocks)
2. An async prefetching pathway for DRAM blocks
3. Integration with the attention kernel to handle streaming attention across tiers

## Deferred Reason

Previously deferred from April 22-24 collect runs due to arXiv HTTP 403. Retrieved via web search snippet on April 24, 2026.
