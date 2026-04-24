---
title: "ReaLB: Real-Time Load Balancing for Multimodal MoE Inference"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.19503
collected: 2026-04-24
tags: [moe, expert-parallelism, load-balancing, multimodal, fp4, vllm, throughput]
---

# ReaLB: Real-Time Load Balancing for Multimodal MoE Inference

**arXiv:** 2604.19503v1  
**Submitted:** April 2026 (very recent)  
**Authors:** Not specified in available search results

## Abstract / Core Claim

Multimodal MoE (MMoE) inference under expert parallelism (EP) suffers from **load imbalance** because vision tokens dominate batch composition during prefill, causing a subset of EP ranks to handle disproportionately heavy workloads. ReaLB introduces **real-time load balancing via dynamic precision adjustment**: ranks with vision-heavy expert assignments run at lower precision (FP4 Tensor Cores) to execute faster, equalizing effective throughput across ranks with zero scheduling overhead.

## Problem

Under EP for multimodal MoE models:
- Vision tokens exhibit higher redundancy along the forward pass compared to text tokens
- During prefill with large batch sizes, vision tokens frequently dominate input sequences
- Under EP, this leads to **severe load imbalance**: some EP ranks get vision-heavy experts and become bottlenecks
- Standard EPLB (Expert Parallel Load Balancing, available in vLLM v0.19.0+) reassigns experts but incurs routing overhead

## Technical Approach

### Dynamic Precision Adjustment Per EP Rank
- At runtime, identify which EP ranks are assigned vision-heavy MoE experts
- For those ranks, switch attention and expert forward passes to FP4 precision (exploiting FP4 Tensor Cores on Blackwell/SM100)
- FP4 ops run faster (higher throughput at lower precision), equalizing wall-clock time across ranks
- Text-heavy ranks remain at higher precision (BF16 or FP8)

### Zero Scheduling Overhead
No expert re-routing or tensor communication overhead introduced. The load balance is achieved purely by adjusting the arithmetic precision per rank, not by moving data or experts.

## Performance Results

- **1.29× layer-level speedup** on multimodal MoE inference
- **≤1.2% accuracy loss** across evaluated models
- Consistent performance gains without accuracy loss

## Models Evaluated

1. Kimi-VL-A3B-Instruct (3B active parameters)
2. Qwen3-VL-30B-A3B-Instruct (30B total, 3B active)
3. ERNIE-4.5-VL-27B-A3B (27B total, 3B active)

All three are recent (2025-2026) multimodal sparse MoE models with ~3B active parameters.

## Implementation

ReaLB is **implemented in vLLM** and demonstrated in production-style evaluation. This makes it the first arXiv-published, vLLM-integrated solution specifically for multimodal MoE EP load balancing.

Hardware requirement: FP4 Tensor Cores (NVIDIA Blackwell SM100 / B200 / GB200). FP4 precision is also available in vLLM v0.20.0 via MXFP4/NVFP4 kernels.

## Relationship to Existing vLLM Work

| Feature | vLLM EPLB (v0.19.0+) | ReaLB |
|---------|---------------------|-------|
| Mechanism | Expert re-routing | Dynamic precision per rank |
| Overhead | Communication overhead (addressed in PR #40730) | Zero |
| Target | Any MoE model | Multimodal MoE specifically |
| Implemented in vLLM | Yes (v0.19.0) | Yes (paper) |
