---
title: "FP8 Quantization"
tags: [quantization, memory, throughput, hardware, mla, deepseek, kernels, jit]
created: 2026-04-14
updated: 2026-04-25
sources: [raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/rocm-optimization.md, raw/2026-04-22-vllm-prs-apr21-22.md, raw/2026-04-25-vllm-prs-apr24-25.md]
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

## MLA + Group FP8 Fusion (PR #38877, merged April 22, 2026)

A kernel-level optimization for DeepSeek V3 and other MLA-based architectures. Fuses per-group dynamic FP8 quantization directly into the MLA attention kernel, eliminating a separate quantization pass.

**Background:** MLA (Multi-head Latent Attention) is DeepSeek's attention design that compresses KV heads through a low-rank projection. Group FP8 applies FP8 quantization per group of heads rather than per tensor, providing finer-grained scale factors. Previously, quantization ran as a separate kernel before attention.

**Performance (B200x4):**
- Output token throughput: 5,345 → 5,367 tokens/second (+0.4%)
- GSM8K accuracy: ~94.8% maintained with fusion

**Also fixes:** NVFP4 pattern matching for `DeepSeek-R1-0528-NVFP4-v2` model variants.

Completes phase 1 of vLLM issue #35792 (full FP8 group quant fusion roadmap for MLA).

(source: raw/2026-04-22-vllm-prs-apr21-22.md)

## Relationship to FP4

FP4 (MXFP4) is the next step below FP8 for model weights, achieving ~4× memory reduction vs BF16. On Blackwell (SM100/SM120), hardware FP4 TensorCores make MXFP4 viable. As of April 2026, vLLM has a new CUTLASS W4A4 MXFP4 MoE kernel (PR #37463) for B200/SM100. See [FP4 Quantization](fp4-quantization.md).

## Humming Quantization Kernel (PR #34556, merged April 24, 2026)

Humming is a **JIT quantization kernel library** from the inclusionAI project, integrated into vLLM as a fourth quantization kernel backend alongside Marlin, CUTLASS, and FlashInfer.

### Format Coverage
- **Weight-activation combos:** W{1–8} A{16/8/4} — the broadest coverage in any single vLLM backend
- **Quantization schemes:** GPTQ, AWQ, FP8, MXFP4, NVFP4, BITNET
- **Exclusions:** GPTQ with `desc_act=True`, HQQ

### Hardware Support
- **Optimal:** Turing (SM75), Ampere (SM80), Ada Lovelace (SM89), Hopper (SM90/SM90a)
- **Partial:** Blackwell datacenter (SM100) and consumer GPUs — functional but not peak-performance

### Online Quantization
Supports serving-time quantization for INT1–INT8 and FP3–FP8 without pre-quantized checkpoints. Bit-depths ≥ 6 recommended for uncalibrated (no calibration dataset) use. Configured via environment variables.

### MoE and CUDA Graph
Supports block-wise MoE operations and is CUDA-graph compatible.

### Benchmarks

| Configuration | Humming | Marlin | Hardware | Note |
|--------------|---------|--------|----------|------|
| W8A16, dense, float16 | ~142 TFLOPS | ~89–90 TFLOPS | H20 | Large matrix dims |

**~1.58× throughput advantage over Marlin** on H20 for W8A16 at large batch sizes.

### Significance
Humming's JIT compilation model makes it the most format-flexible backend currently in vLLM. Unlike Marlin (W4/W8 GPTQ/AWQ), CUTLASS (FP8/FP4/MoE), and FlashInfer (attention-kernel-adjacent), Humming can serve sub-INT8 precision (W1–W3) and FP3–FP7 without separate kernel contributions. On older GPU generations (Ampere, Turing) that lack native FP4 TensorCores, Humming's JIT path is likely the best route to sub-INT8 inference.

(source: raw/2026-04-25-vllm-prs-apr24-25.md)

## Open Questions
- How does FP8 KV cache interact with prefix caching quality?
- What's the accuracy degradation for FP4 on reasoning-heavy tasks?
- What does phase 2 of issue #35792 (beyond group FP8) entail for MLA + quantization fusion?
- What is Humming's throughput vs Marlin on Hopper (SM90) for W4A16 (typical GPTQ config)?
- Does Humming's JIT compilation add measurable first-request latency vs pre-compiled Marlin kernels?
- Does Humming support FP8 KV cache quantization, or only weight/activation quantization?
