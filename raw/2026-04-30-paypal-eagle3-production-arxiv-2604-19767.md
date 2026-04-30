---
title: "Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.19767
collected: 2026-04-30
tags: [speculative-decoding, eagle3, production, vllm, h100, nemotron, throughput, latency]
---

# Accelerating PayPal's Commerce Agent with Speculative Decoding (arXiv 2604.19767)

## Overview

Empirical production study evaluating EAGLE3 speculative decoding applied to PayPal's Commerce Agent — a fine-tuned `llama3.1-nemotron-nano-8B-v1` model serving real e-commerce inference workloads. Deployed via vLLM and benchmarked against NVIDIA NIM on identical 2×H100 hardware.

## Experimental Setup

- **Model**: fine-tuned llama3.1-nemotron-nano-8B-v1 (PayPal's commerce SLM)
- **Hardware**: 2×H100 (identical for both vLLM+EAGLE3 and NIM baselines)
- **Framework**: vLLM with EAGLE3 speculative decoding
- **Configurations tested**: 40 total — gamma ∈ {3, 5}, concurrency ∈ {1, 2, 4, 8, 16, 32}, temperature ∈ {0.0, 0.5}

## Key Results

### Throughput and Latency (gamma=3)
- **22–49% throughput improvement** vs NVIDIA NIM (same 2×H100)
- **18–33% latency reduction** vs NVIDIA NIM

### Token Acceptance Rate
- gamma=3: **~35.5% acceptance rate** — stable across all concurrency levels (1–32) and temperatures (0, 0.5)
- gamma=5: diminishing returns — **~25% acceptance rate**; gamma=5 yields lower ROI than gamma=3 for this workload

### GPU Cost Reduction
- Single H100 with EAGLE3 spec decode **matches or exceeds** 2×H100 NIM in throughput
- Implication: **50% GPU cost reduction** — halve hardware while maintaining service level

### Output Quality
- LLM-as-Judge evaluation confirms **fully preserved output quality** at both gamma=3 and gamma=5
- No measurable quality degradation from speculative decoding on the commerce agent task

## Key Observations

1. **Acceptance rate stability**: 35.5% held constant across all concurrency levels tested — spec decode scales linearly in throughput gain as concurrency increases, unlike some prior reports of diminishing returns at high concurrency for other workloads
2. **gamma=3 optimal for this workload**: The commerce agent generates predictable structured outputs (product queries, structured responses), which explains the consistent 35.5% rate vs the lower 25% at gamma=5 (longer speculation overshoots the predictable zone)
3. **Comparison to NIM**: NIM is NVIDIA's optimized closed inference product; beating it 22–49% on identical hardware demonstrates that open serving frameworks (vLLM) with speculative decoding can outperform vendor-optimized closed solutions on fine-tuned models

## Scope Note

Study applies to fine-tuned SLMs (8B) in production e-commerce. Acceptance rates will differ for other domains (reasoning, code) and model sizes. The 35.5% rate is specific to the PayPal commerce agent's output distribution.

## Citation

PayPal AI, arXiv 2604.19767, April 2026.
