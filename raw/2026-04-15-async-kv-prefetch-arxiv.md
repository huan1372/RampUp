---
title: "Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching"
source_url: https://arxiv.org/abs/2504.06319
collected: 2026-04-15
type: arxiv-paper
arxiv_id: 2504.06319
---

# Asynchronous KV Cache Prefetching for LLM Inference

## Abstract
Proposes an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap.

## Problem
Memory bandwidth is a primary bottleneck in LLM inference attention computation. HBM access latency for KV cache dominates inference time.

## Key Contribution
Strategically schedule KV Cache prefetching operations to coincide with periods of active computation, leveraging otherwise idle memory bandwidth to proactively load KV Cache into the faster L2 cache layer before it's needed.

### Core Mechanism
- **During computation**: idle memory bandwidth is used to prefetch upcoming KV cache entries into GPU L2 cache
- **At access time**: L2 cache hits mask the HBM access latency
- **Result**: effectively hides memory latency behind computation

## Results (NVIDIA H20 GPU)
- **2.15× improvement** in attention kernel efficiency
- **Up to 1.97× end-to-end throughput** improvement
- Performance **surpasses FlashAttention-3 baseline**

## Compatibility
Maintains orthogonality with existing optimization techniques — can be layered on top of FlashAttention, PagedAttention, etc. without conflict.

## Relevance to vLLM
This approach is complementary to vLLM's existing KV cache management. Could be integrated with the KV connector framework or applied within the attention kernel layer. The H20 is a common production GPU (common in Chinese cloud deployments due to export controls), making this result operationally significant.
