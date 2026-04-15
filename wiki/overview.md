---
title: "Overview & Synthesis"
tags: [overview, synthesis, meta]
created: 2026-04-14
updated: 2026-04-15
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/vllm-releases.md]
related: [concepts/paged-attention.md, concepts/model-runner-v2.md]
---

# Inference Optimization — Overview & Synthesis

## The Landscape (April 2026)

LLM inference optimization has converged on a core set of techniques that work together: efficient memory management (PagedAttention), dynamic scheduling (continuous batching), and hardware-aware computation (quantization, parallelism). The three dominant open-source serving engines are **vLLM**, **SGLang**, and **TensorRT-LLM**, each with different strengths.

## vLLM: Current State

vLLM (v0.19.0 as of April 2026) has become the most widely adopted open-source inference engine. Key recent developments:

- **Model Runner V2 (MRV2)** — a ground-up rewrite of the execution core; opt-in in v0.18.0 (March 2026), **default in v0.19.0** (April 3, 2026). Cleaner, more modular, GPU-native input preparation, async-first. Delivers 56% throughput gain on GB200 from input prep alone.
- **V1 Engine** — the default since v0.8.0, delivering 1.7x throughput over the original engine. Prefix caching is now nearly free (<1% overhead at 0% hit rate).
- **Blackwell support** — full SM120 support as of v0.15.1 (Feb 2026), including NVFP4 MoE kernels.
- **Compilation** — moving toward `torch.compile` as the default optimization path, with custom Helion kernels planned.

## Competitive Positioning

Based on Clarifai benchmarks (GPT-OSS-120B on 2x H100):
- **vLLM**: highest throughput at high concurrency (4,741 tok/s at 100 requests), fastest TTFT
- **SGLang**: most stable inter-token latency (4-21ms), strong RadixAttention for multi-turn
- **TensorRT-LLM**: best single-request throughput, but scales worse and requires compilation step

## Key Optimization Vectors

1. **Memory** — PagedAttention, KV cache offloading to CPU, FP8/FP4 quantization
2. **Compute** — speculative decoding, continuous batching, chunked prefill, fused kernels
3. **Scale** — tensor/pipeline/expert parallelism, disaggregated prefill-decode, elastic serving
4. **Scheduling** — DBO (Dual-Batch Overlap), async scheduling with zero-bubble overlap

## Open Questions

- How does MRV2 performance compare to MRV1 across the model zoo? (MRV1 still handles "long tail" cases)
- What's the practical impact of torch.compile on cold-start times?
- How does vLLM's CPU KV cache offloading compare to SGLang's approach?
