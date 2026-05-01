---
title: "FP8 Quantization"
tags: [quantization, memory, throughput, hardware, mla, deepseek, kernels, jit, vit, multimodal, blackwell, async-tp, torch-compile]
created: 2026-04-14
updated: 2026-05-01
sources: [raw/vllm-benchmarks-2026.md, raw/vllm-releases.md, raw/rocm-optimization.md, raw/2026-04-22-vllm-prs-apr21-22.md, raw/2026-04-25-vllm-prs-apr24-25.md, raw/2026-04-27-vllm-prs-apr26-27.md, raw/2026-05-01-vllm-prs-may1.md]
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

## FP8 Attention for Vision Transformer (ViT) Encoders (PR #38065, April 27, 2026)

Multimodal workloads bottleneck on the ViT encoder once the language model is quantized. PR #38065 extends FP8 to ViT encoder attention via the FlashInfer cuDNN backend.

### Scope
- **Supported models:** Qwen3 VL family: `qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`, `qwen3_5_moe`
- **Hardware requirement:** SM90+ (Hopper or Blackwell); no-op fallback on older GPUs
- **cuDNN minimum:** version 9.17.1
- **Scaling:** dynamic by default; static calibration optional for production

### Performance (Qwen3-VL-30B, GB200, 3 images/request)

E2E encoder forward time speedup:

| Resolution | Speedup |
|------------|---------|
| HD (720×1280) | 0.87× (regression) |
| FullHD (1080×1920) | 0.99× (neutral) |
| QHD (1440×2560) | 1.08× |
| 4K (2160×3840) | **1.18×** |

Core cuDNN kernel (head_dim=128, seq_len=8192):

| Hardware | Speedup |
|----------|---------|
| GB200 | 1.12× |
| GB300 | **1.42×** |

**Key pattern:** FP8 ViT attention only benefits at high resolution (long ViT sequence lengths). At HD, FP8 conversion overhead exceeds kernel speedup — configurable, not forced.

(source: raw/2026-04-27-vllm-prs-apr26-27.md)

## Faster Per-Token FP8 Group Quant Packed Kernel for Blackwell (PR #41326, May 1, 2026)

Introduces a FP8 per-token group quantization kernel optimized specifically for Blackwell (SM100/SM120) GPUs. The "packed" design fuses the FP8 quantization step with the subsequent GEMM dispatch, reducing separate kernel launch overhead and HBM round-trip traffic vs. the prior separate-kernel sequence.

**Per-token group quantization:** Applies a distinct FP8 scale factor per token per group of channels. This is finer-grained than per-tensor FP8 (one scale per entire tensor) — providing accuracy closer to FP16 while retaining FP8 compute throughput. The additional scale factors add minor memory overhead but negligible compute cost on Blackwell's native FP8 TensorCores.

**"Packed" kernel:** The FP8 quantization and scale-factor computation are fused into the same CUDA kernel as the quantized GEMM, eliminating the intermediate store of quantized weights to HBM and reducing memory bandwidth pressure.

**Scope:** Blackwell-specific (SM100/SM120). H100/H200 (SM90) fallback behavior is unchanged.

**Relationship to existing FP8 work:**
- Complements PR #38877 (MLA + Group FP8 fusion, April 22): that PR fused FP8 quant into MLA attention; this PR targets the general GEMM path (non-MLA linear layers)
- Part of the broader FP8 group GEMM roadmap (vLLM issue #35792)

(source: raw/2026-05-01-vllm-prs-may1.md)

## FlashInfer FP8 Async TP Fusion (PR #39505, May 1, 2026)

Adds a `torch.compile`-level fusion pass that combines FlashInfer FP8 GEMM operations with async tensor parallelism (TP) allreduce collectives, and preserves allreduce fusion ordering in the compiler.

**Problem (GitHub issue #27985):** Async TP overlaps communication with computation by launching allreduce ops asynchronously while the next layer's computation begins. The FP8 quantization step must precede the allreduce. `torch.compile`'s fusion passes could reorder these ops, breaking the async TP overlap schedule or producing incorrect outputs for FP8 models at TP≥2.

**Fix:** Two changes:
1. Adds a fusion pattern that recognizes FlashInfer FP8 GEMM → allreduce sequences and correctly fuses them without reordering
2. Adds compiler-level ordering constraints that preserve allreduce position relative to FP8 ops even when other fusion passes run

**Scope:** Affects FP8-quantized models served with TP≥2 via `torch.compile`. Most impactful on Blackwell B200/GB200 multi-GPU deployments where async TP is the default communication pattern.

**Relationship to tensor parallelism page:** Also documented at [Tensor Parallelism](tensor-parallelism.md#flashinfer-fp8-async-tp-fusion-pr-39505-may-1-2026).

(source: raw/2026-05-01-vllm-prs-may1.md)

## Open Questions
- How does FP8 KV cache interact with prefix caching quality?
- What's the accuracy degradation for FP4 on reasoning-heavy tasks?
- What does phase 2 of issue #35792 (beyond group FP8) entail for MLA + quantization fusion?
- What is Humming's throughput vs Marlin on Hopper (SM90) for W4A16 (typical GPTQ config)?
- Does Humming's JIT compilation add measurable first-request latency vs pre-compiled Marlin kernels?
- Does Humming support FP8 KV cache quantization, or only weight/activation quantization?
- Will FP8 ViT attention support be extended to non-Qwen3 multimodal models (e.g., LLaVA, InternVL)?
- Does FP8 ViT attention interact with prefix caching on the vision token side?
- Does PR #41326's packed kernel deliver measurable TPOT improvement over the prior separate-kernel path? (No benchmark numbers in the PR as captured)
- Does PR #39505's allreduce fusion ordering constraint add compiler overhead (longer compile time) for models that don't use FP8 + async TP together?
