---
title: "HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.18529
collected: 2026-04-24
tags: [kv-cache, cpu-gpu, hybrid, long-context, cxl, offloading, memory, attention]
---

# HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing

**arXiv:** 2604.18529  
**Submitted:** April 20, 2026  
**Authors:** Mao Lin, Xi Wang, Guilherme Cox, Dong Li, Hyeran Jeon

## Abstract / Core Claim

As LLMs support millions of tokens, KV caches grow to hundreds of gigabytes — beyond single-GPU memory capacity. Existing solutions (KV cache pruning and offloading) either lose information or underutilize hardware by using either GPU or CPU for attention exclusively. HybridGen enables **CPU-GPU collaborative attention** on systems with CXL-expanded tiered memory, splitting attention computation between CPU and GPU simultaneously to exploit both compute resources and reduce per-device memory pressure.

## Technical Design

### 1. Attention Logit Parallelism
Split attention computation by partitioning the KV cache across CPU and GPU. Each device computes attention logits for its portion of the KV cache independently. Results are combined via log-sum-exp aggregation (standard flash-attention reduction). This is analogous to tensor parallelism but applied to the KV dimension across CPU and GPU rather than across multiple GPUs.

### 2. Feedback-Driven Scheduler
A runtime scheduler monitors per-device compute utilization and adjusts the CPU/GPU split dynamically. If GPU becomes a bottleneck, less KV is shifted to GPU; if CPU is bottlenecked, more KV is handled on GPU. This prevents load imbalance from degrading throughput.

### 3. Semantic-Aware KV Cache Mapping
KV cache blocks are placed in memory (GPU HBM or CXL DRAM) according to their **semantic importance** to the current query. High-importance KV blocks stay in GPU HBM for fast attention; low-importance blocks are mapped to CPU-accessible CXL DRAM. Importance is estimated from attention scores during earlier decode steps.

## Hardware Setup

Evaluated on systems with CXL-expanded memory (CPU-accessible DRAM pool reachable via CXL interface). Three GPU platforms tested. CXL allows DRAM to be addressable by both CPU and GPU without explicit data copies in some configurations.

## Performance Results

- **1.41×–3.2× throughput improvement** over 6 state-of-the-art KV cache management baselines on average
- Maintains superior accuracy vs baselines (KV pruning/eviction loses accuracy; HybridGen doesn't drop KV blocks)
- Evaluated on 3 LLM models across 11 different model sizes and 3 GPU platforms

## Baselines Compared

Six existing KV cache management methods (details of exact baselines not available from search snippet; likely include H2O, SnapKV, StreamingLLM, and other KV eviction/pruning approaches).

## Key Distinction from Offloading

Standard KV offloading moves KV blocks to CPU and retrieves them serially (blocking). HybridGen computes attention **on the CPU-resident KV concurrently** with GPU attention, exploiting both compute units. The CPU is not just a storage medium — it is an active compute participant.

## Relationship to vLLM

No vLLM integration confirmed. Integration would require:
1. Extending the KV cache block table to differentiate HBM vs CXL-DRAM blocks
2. A dual-device attention kernel (GPU path + CPU path with aggregation)
3. A scheduler aware of CPU memory bandwidth and compute capacity

## Deferred Reason

Previously deferred from April 20-24 collect runs due to arXiv HTTP 403. Retrieved via web search snippet on April 24, 2026.
