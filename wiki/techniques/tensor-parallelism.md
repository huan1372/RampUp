---
title: "Tensor Parallelism"
tags: [parallelism, multi-gpu, scale]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/redhat-tuning.md]
related: [techniques/disaggregated-serving.md, concepts/model-runner-v2.md, techniques/fp8-quantization.md]
---

# Tensor Parallelism

## Summary
Tensor parallelism (TP) splits individual model layers across multiple GPUs, so each GPU computes a slice of every layer's matrix operations. This is the primary strategy for serving models too large for a single GPU, and the most common multi-GPU parallelism mode in vLLM.

## Problem It Solves
Large models (70B+) don't fit in a single GPU's memory. Even if they fit, a single GPU may not deliver sufficient throughput. TP distributes both the compute and the memory across GPUs.

## How It Works
1. Weight matrices in attention and MLP layers are sharded column-wise or row-wise across GPUs
2. Each GPU computes its portion of the forward pass independently
3. All-reduce communication synchronizes results between GPUs after each layer
4. The KV cache is also distributed — each GPU stores its portion

TP requires fast inter-GPU communication (NVLink preferred). The communication overhead grows with the number of GPUs, so there's a sweet spot.

## Implementation in vLLM
- `--tensor-parallel-size N` (or `-tp N`)
- MRV2 supports piecewise CUDA graphs for pipeline parallelism (PR #35162)
- Async TP is being enabled by default (Q2 2026 roadmap)
- Expert parallelism (EP) available for MoE models, with EPLB for load balancing

## GPU Configuration Strategy
From Red Hat's tuning guide — for a fixed GPU budget:
1. Find the minimum GPUs needed to load the model with sufficient KV cache
2. Deploy maximum replicas with that minimum count
3. Test with more GPUs per replica (fewer replicas) and compare throughput
4. Example with 8× H100: test 8×TP1 (8 replicas), 4×TP2, 2×TP4, 1×TP8

## Trade-offs
- More GPUs per replica = more memory per model, but more communication overhead and fewer replicas
- NVLink is critical — PCIe interconnects severely limit TP scaling
- TP works best up to ~8 GPUs; beyond that, pipeline parallelism or expert parallelism may be better

## When to Use
- Any model that doesn't fit on a single GPU
- When latency matters more than cost (more GPUs = lower per-request latency)
- Models > 13B parameters generally benefit from TP2+

## Open Questions
- What's the throughput crossover point between TP4×2 replicas vs TP8×1 replica for 70B models?
- How does Async TP change the scaling characteristics?
