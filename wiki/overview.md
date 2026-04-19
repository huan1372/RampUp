---
title: "Overview & Synthesis"
tags: [overview, synthesis, meta]
created: 2026-04-14
updated: 2026-04-19
sources: [raw/vllm-roadmap-q2-2026.md, raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/2026-04-14-vllm-rampup-recap.md, raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md, raw/2026-04-19-calibrated-speculative-decoding-arxiv.md]
related: [concepts/paged-attention.md, concepts/model-runner-v2.md, concepts/continuous-batching.md, concepts/chunked-prefill.md]
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

1. **Memory** — PagedAttention, KV cache offloading to CPU, FP8/FP4 quantization, sub-FP8 KV compression (TurboQuant: 2.6–4.9×, merged Apr 2026; WHT overhead reduced Apr 2026)
2. **Compute** — speculative decoding (P-EAGLE 1.55-1.69×; CSD 2.33× peak, Apr 2026), continuous batching, chunked prefill, fused kernels; MXFP4 W4A4 CUTLASS MoE kernel for B200 (Apr 2026)
3. **Scale** — tensor/pipeline/expert parallelism, disaggregated prefill-decode, elastic serving
4. **Scheduling** — DBO (Dual-Batch Overlap), async scheduling with zero-bubble overlap; multimodal scheduler overhead reduction (Apr 2026)

### KV Cache Compression: Expanding Beyond FP8 (April 2026)

TurboQuant (PR #38479, merged April 15, 2026) extends vLLM's KV cache compression below FP8 for the first time. Using WHT rotation on keys and uniform quantization on values, it achieves 2.6–4.9× compression ratios at the cost of higher compute overhead and model-dependent quality risk. The conservative `turboquant_k8v4` preset (FP8 keys, 4-bit values) delivers TPOT improvement on long-context workloads with modest throughput overhead. Aggressive 3-bit compression shows severe quality degradation and requires validation. PR #40194 (April 2026) removes a redundant random sign flip from the WHT pipeline, reducing per-token overhead. See [KV Cache Quantization](techniques/kv-cache-quantization.md). (source: raw/2026-04-16-turboquant-kv-compression-pr38479.md, raw/2026-04-19-vllm-prs-apr17-19.md)

### MXFP4 W4A4 for Blackwell MoE: Pushing to 4-Bit Activations (April 2026)

PR #37463 (merged April 2026) adds a CUTLASS-based W4A4 MXFP4 MoE kernel targeting SM100 (Blackwell B200). This is the first CUTLASS alternative to the FlashInfer MXFP4 MoE backend, exploiting SM100 native FP4 TensorCores and TMA async memory loads. W4A4 quantizes both weights and activations to MXFP4 format, maximizing MoE expert GEMM throughput on B200 at the cost of higher accuracy risk. Specific benchmark numbers vs FlashInfer not yet published. See [FP4 Quantization](techniques/fp4-quantization.md). (source: raw/2026-04-19-vllm-prs-apr17-19.md)

### Calibrated Speculative Decoding: Reducing False Rejections (April 2026)

arXiv 2604.13634 (April 15, 2026) proposes CSD, a training-free addition to speculative decoding that addresses false rejections — cases where a semantically valid draft token is discarded for being lexically different from the target distribution's top token. CSD adds Online Correction Memory (tracks historical rejections to propose rescue candidates) and Semantic Consistency Gating (validates candidates via probability ratio). Peak 2.33× throughput speedup. Not yet integrated into vLLM. See [Speculative Decoding](techniques/speculative-decoding.md). (source: raw/2026-04-19-calibrated-speculative-decoding-arxiv.md)

## Glossary (quick reference)

- **TTFT** — Time To First Token. Latency from prompt submission to first response token. Measures prefill speed. Target: <200ms.
- **ITL** — Inter-Token Latency. Time between consecutive output tokens during streaming. Measures decode speed. Target: <30ms.
- **KV cache** — Stored key-value tensors from attention for all tokens seen so far. See [KV Cache Management](concepts/kv-cache-management.md).
- **Prefill** — Processing input prompt (parallel, compute-bound).
- **Decode** — Generating output tokens one at a time (sequential, memory-bandwidth-bound).
- **Preemption** — Evicting a request's KV blocks when memory runs out (swap to CPU or recompute). See [PagedAttention](concepts/paged-attention.md).
- **TP** — Tensor Parallelism. Splitting model layers across multiple GPUs. See [Tensor Parallelism](techniques/tensor-parallelism.md).

(source: raw/2026-04-14-vllm-rampup-recap.md)

## Open Questions

- How does MRV2 performance compare to MRV1 across the model zoo? (MRV1 still handles "long tail" cases)
- What's the practical impact of torch.compile on cold-start times?
- How does vLLM's CPU KV cache offloading compare to SGLang's approach?
