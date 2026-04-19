---
title: "FP8 Quantization"
tags: [quantization, memory, throughput, hardware]
created: 2026-04-14
updated: 2026-04-19
sources: [raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/rocm-optimization.md]
related: [concepts/kv-cache-management.md, techniques/tensor-parallelism.md, techniques/kv-cache-quantization.md, techniques/fp4-quantization.md]
---

# FP8 Quantization

## Summary
FP8 (8-bit floating point) quantization halves model memory footprint with minimal accuracy loss. On hardware with native FP8 tensor cores (H100, H200, B200), it is effectively free — delivering up to 1.6x throughput gain with near-zero quality degradation.

## Problem It Solves
Large LLMs require enormous GPU memory just for weights, leaving less room for KV cache and thus limiting batch size and throughput. FP8 reduces weight memory by 2x compared to FP16/BF16.

## How It Works
Two modes:
1. **Dynamic FP8** — vLLM handles quantization at serving time, no model conversion needed. Lower throughput than static but zero setup cost.
2. **Static FP8** — pre-quantize weights offline using tools like AMD Quark or NVIDIA TensorRT Model Optimizer. Higher throughput because scaling factors are pre-computed.

Also applies to KV cache: setting `kv_cache_dtype=fp8` halves KV cache memory, effectively doubling the number of concurrent sequences.

## Implementation in vLLM
- Dynamic FP8: just serve the model, vLLM auto-quantizes on H100+
- Static FP8: pre-quantize, then serve normally
- KV cache FP8: `--kv-cache-dtype fp8`
- vLLM also supports NVFP4 for even more aggressive compression on Blackwell GPUs

## Benchmarks
| Metric | FP16 | FP8 | Hardware | Notes |
|--------|------|-----|----------|-------|
| Memory | 1x   | 0.5x | H100 | Weight memory |
| Throughput | 1x | up to 1.6x | H100 | With static FP8 |
| Accuracy | baseline | minimal loss | — | Model-dependent |

## Trade-offs
- Dynamic FP8: zero setup but slightly lower throughput than static
- Static FP8: requires offline quantization step but better runtime performance
- FP4 (NVFP4): 4x memory reduction but more accuracy risk, only on Blackwell

## When to Use
- Always on H100/H200/B200 hardware — it's nearly free
- Especially valuable when KV cache memory is the bottleneck (long contexts, high concurrency)
- Skip on older hardware without native FP8 tensor cores

## Relationship to KV Cache Quantization

FP8 is the lowest-overhead point on the KV cache compression spectrum. Sub-FP8 approaches (TurboQuant at 2–4 bits, merged into vLLM main April 15, 2026) achieve 2.6–4.9× compression but carry higher compute overhead and quality risk. See [KV Cache Quantization](kv-cache-quantization.md) for the full spectrum.

## Relationship to FP4

FP4 (MXFP4) is the next step below FP8 for model weights, achieving ~4× memory reduction vs BF16. On Blackwell (SM100/SM120), hardware FP4 TensorCores make MXFP4 viable. As of April 2026, vLLM has a new CUTLASS W4A4 MXFP4 MoE kernel (PR #37463) for B200/SM100. See [FP4 Quantization](fp4-quantization.md).

## Open Questions
- How does FP8 KV cache interact with prefix caching quality?
- What's the accuracy degradation for FP4 on reasoning-heavy tasks?
