---
title: "Tensor Parallelism"
tags: [parallelism, multi-gpu, scale, moe, expert-parallelism, load-balancing, eplb]
created: 2026-04-14
updated: 2026-04-25
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/redhat-tuning.md, raw/2026-04-24-realb-moe-arxiv.md, raw/2026-04-25-vllm-prs-apr24-25.md]
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

## Expert Parallelism (EP) and Load Balancing

For MoE models, expert parallelism (EP) distributes distinct experts across GPUs so each GPU holds and runs a subset of experts. Unlike TP, EP requires token routing via the MoE gating mechanism.

**vLLM support**: EP available since v0.19.0 with EPLB (Expert Parallel Load Balancing) for static routing optimization. PR #40730 (v0.20.0) removed asyncio overhead from EPLB communication.

### ReaLB: Real-Time Load Balancing for Multimodal MoE (arXiv 2604.19503, April 2026)

Under EP for **multimodal** MoE models, vision tokens dominate during prefill, creating severe load imbalance: some EP ranks get vision-heavy expert assignments and become bottlenecks. Standard EPLB re-routes experts (adds communication overhead); ReaLB solves the imbalance by adjusting **computation precision** per rank at runtime.

**Mechanism**: For EP ranks with vision-heavy expert assignments, switch forward passes to FP4 (faster at lower precision on Blackwell SM100 Tensor Cores). Text-heavy ranks remain at BF16/FP8. Wall-clock time equalizes across ranks with zero routing overhead.

**Results (multimodal MoE models)**:
- **1.29× layer-level speedup**
- **≤1.2% accuracy loss**
- Evaluated: Kimi-VL-A3B-Instruct, Qwen3-VL-30B-A3B-Instruct, ERNIE-4.5-VL-27B-A3B
- **Implemented in vLLM**

| Feature | vLLM EPLB (v0.19.0+) | ReaLB (research) |
|---------|---------------------|-----------------|
| Mechanism | Expert re-routing | Per-rank precision |
| Overhead | Communication overhead | Zero |
| Target | Any MoE | Multimodal MoE |

(source: raw/2026-04-24-realb-moe-arxiv.md)

## EPLB Replica Selection Bias Fix (PR #40810, April 24, 2026)

**Problem:** The EPLB fused MoE router had a hash-collision bug when `top_k` (active experts per token) was a multiple of the replica count. All tokens for affected expert groups mapped to the same replica via the hash function, causing load imbalance exceeding 90% in adversarial configurations — effectively negating the load-balancing purpose.

**Fix:** Replaced the hash function with a **Knuth multiplicative hash** for replica selection. Knuth's multiplicative method distributes keys uniformly across buckets regardless of the algebraic relationship between key values and bucket count, breaking the periodicity that caused collapse.

**Results (Qwen3.5-A17B on 8× B200):**

| Metric | Before | After |
|--------|--------|-------|
| Max/mean workload ratio | 1.2 (with >90% imbalance in extreme cases) | 1.07 |

A regression test `test_eplb_map_hot_expert_replica_balance` was added (496 tests passing).

**Impact:** Correctness and performance fix for any deployment using EP ≥ 2 with `top_k` values that are multiples of the replica count. Common configurations such as `top_k=2` with 2 replicas and `top_k=8` with 4 replicas are affected. Without the fix, EPLB silently degrades to near-worst-case distribution for those configurations.

(source: raw/2026-04-25-vllm-prs-apr24-25.md)

## Open Questions
- What's the throughput crossover point between TP4×2 replicas vs TP8×1 replica for 70B models?
- How does Async TP change the scaling characteristics?
- Does ReaLB's per-rank FP4 assignment require Blackwell SM100 or does it work on H100 (SM90) with MXFP8 fallback?
- How does ReaLB interact with vLLM's existing EPLB? Can they compose (EPLB for coarse routing, ReaLB for fine-grained precision adjustment)?
- Does the Knuth multiplicative hash uniformly fix imbalance for all `top_k` / replica count ratios, or are there remaining edge cases at very large EP counts?
