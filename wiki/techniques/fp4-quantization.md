---
title: "FP4 / MXFP4 Quantization"
tags: [quantization, memory, throughput, hardware, blackwell, hopper, amd, moe, fp4, mxfp4, w4a4, nvfp4, emulation]
created: 2026-04-19
updated: 2026-04-23
sources: [raw/2026-04-19-vllm-prs-apr17-19.md, raw/vllm-benchmarks-2026.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-23-vllm-prs-apr22-23.md]
related: [techniques/fp8-quantization.md, techniques/kv-cache-quantization.md, techniques/tensor-parallelism.md, concepts/kv-cache-management.md]
---

# FP4 / MXFP4 Quantization

## Summary

FP4 (4-bit floating point) quantization achieves ~4× weight memory reduction compared to BF16 at the cost of more accuracy risk than FP8. The microscaling variant (MXFP4) uses per-group scale factors to mitigate outlier sensitivity. On NVIDIA Blackwell GPUs (SM120, SM100), FP4 quantization is hardware-native; CUDA cores include FP4 TensorCores that make MXFP4 GEMMs substantially faster than software-emulated 4-bit. As of April 2026, vLLM supports MXFP4 for both weight-only and W4A4 (4-bit weights + 4-bit activations) configurations, primarily targeting MoE models on Blackwell.

## Problem It Solves

For very large MoE models (70B+, 120B+), even FP8 weight memory can exceed a single-node GPU cluster's VRAM. FP4 cuts weight memory by another 2× vs FP8 (4× vs BF16), enabling larger models to fit without multi-node tensor parallelism. For MoE expert GEMMs specifically, FP4 TensorCores on Blackwell deliver dramatically higher TFLOPS than FP8, because expert computation is the dominant compute bottleneck during decode.

## Formats

| Format | Bits | Scaling | vLLM flag | Hardware |
|--------|------|---------|-----------|----------|
| NVFP4 (weight-only, W4A16) | 4W / 16A | per-group | `--quantization nvfp4` | SM120, SM100 |
| MXFP4 (W4A4) | 4W / 4A | microscaling | CUTLASS MoE kernel (PR #37463) | SM100 (B200) |

- **NVFP4** (NVIDIA FP4): weights stored in 4-bit, activations remain FP8/BF16. Supported since vLLM v0.15.1 (SM120) and earlier for SM100.
- **MXFP4 W4A4**: both weights AND activations are quantized to 4-bit at compute time. Only viable on SM100 (Blackwell B200 and higher) where hardware FP4 TensorCores handle both operands natively.

(source: raw/2026-04-19-vllm-prs-apr17-19.md)

## MXFP4 Format Details

MXFP4 is the MX (microscaling) consortium standard for 4-bit floating point:
- E2M1 encoding: 1 sign bit, 2 exponent bits, 1 mantissa bit
- Per-group (block) scale factor: one FP8 or FP16 scale per 16 or 32 elements
- The group scale factor dramatically extends the dynamic range compared to naïve 4-bit, making it viable for weights with outlier channels
- On SM100, the FP4 TensorCore instructions (`mma.sync.aligned.m16n8k64.f32.e2m1.e2m1.f32`) operate natively on MXFP4 format

## CUTLASS MoE Kernel for SM100 (PR #37463, merged April 2026)

The most significant FP4 development in the April 17-19, 2026 window:

**What it adds**:
- A CUTLASS-based W4A4 MXFP4 MoE kernel targeting SM100 (Blackwell B200)
- CUTLASS (CUDA Templates for Linear Algebra, NVIDIA) generates highly optimized GEMM code using SM100 WGMMA (Warpgroup Matrix Multiply-Accumulate) instructions
- `ThreadBlockShape = Shape<_128, _128, _128>` — tuned for SM100 warpgroups
- TMA (Tensor Memory Accelerator) integration for async global→shared memory loads — reduces memory latency vs. non-TMA loads
- Targets MoE expert GEMMs specifically: the `topk_ids` routing dispatch is fused into the CUTLASS template

**Why CUTLASS vs FlashInfer**:
- FlashInfer MXFP4 MoE kernel was the prior default on Blackwell
- Hugging Face/apsys TFLOPS benchmarks showed a meaningful gap between vLLM/FlashInfer and optimized CUTLASS backends for FP4 MoE on B200
- CUTLASS templates allow deeper SM100-specific tuning (warpgroup scheduling, TMA pipelining) not yet available in FlashInfer's FP4 path

(source: raw/2026-04-19-vllm-prs-apr17-19.md)

## Implementation in vLLM

### Weight-Only NVFP4 (W4A16)
- Available since v0.15.1 for SM120 (RTX 5000/6000 Blackwell) and v0.16.0 for SM100 (B200)
- Flag: `--quantization nvfp4`
- Checkpoint format: NF4 checkpoints from ModelOpt or llm-compressor
- MoE support: FlashInfer MXFP4+MXFP8 backend, enable with `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`

### NVFP4 MoE Emulation Fallback for H100/MI300/MI350 (PR #35737, merged April 22, 2026)

Prior to this PR, NVFP4-quantized checkpoints could only run on Blackwell hardware. PR #35737 adds a Triton-based software emulation backend that handles FP4 quantize/dequantize on non-Blackwell hardware:

- **Supported hardware:** NVIDIA H100 (SM90 / Hopper), AMD MI300, AMD MI350
- **Mechanism:** Triton `TritonExperts` kernel emulates FP4 operations when hardware FP4 TensorCores are unavailable. `quark_moe.py` refactored to standardize on `TritonExperts` for OCP MX quantization emulation
- **Selection:** Automatic — hardware detection routes to emulation if native NVFP4 kernels are absent; no user flag needed
- **Accuracy (WikiText, Qwen3-30B-A3B-NVFP4):** word perplexity 10.90 (BF16) → 11.26 (NVFP4 emulation); ~3.3% relative degradation. PIQA accuracy 0.79–0.80 (stable)
- **Trade-off:** Memory footprint reduced (NVFP4 checkpoint vs BF16), but compute is slower than Blackwell native because emulation runs scalar FP4 ops without hardware acceleration

(source: raw/2026-04-23-vllm-prs-apr22-23.md)

### MXFP4 W4A4 CUTLASS MoE (as of April 2026)
- Merged in PR #37463 to vLLM main
- Not yet in a numbered release as of 2026-04-19; requires building from source
- Targets SM100 only (B200/GB200); SM120 (DGX Spark, RTX PRO 6000) uses a different SM120 path
- Activated automatically when MXFP4 W4A4 checkpoints are loaded on SM100 hardware

## Benchmarks

| Configuration | Hardware | Notes |
|---------------|----------|-------|
| DeepSeek R1/V3 NVFP4 W4A16 | DGX B200 (NVL72) | 2.2k tok/s/H200 at Wide-EP (vLLM blog Dec 2025) |
| gpt-oss-120B MXFP4 MoE | GB200 NVL72 | Subject of Feb 2026 vLLM blog; FlashInfer backend; CUTLASS kernel (PR #37463) expected to improve further |

Specific W4A4 CUTLASS vs FlashInfer benchmark numbers not yet available for PR #37463 (merged to main, no official benchmark published as of 2026-04-19).

(source: raw/2026-04-19-vllm-prs-apr17-19.md, raw/vllm-benchmarks-2026.md)

## Trade-offs

| | NVFP4 W4A16 (Blackwell native) | NVFP4 Emulation (H100/MI300/MI350) | MXFP4 W4A4 |
|--|--------------------------------|-------------------------------------|------------|
| Memory savings | ~4× vs BF16 weights | ~4× vs BF16 weights | ~4× weights + ~4× activation memory during compute |
| Hardware requirement | SM120 or SM100 | H100 (SM90), MI300, MI350 | SM100 only |
| Quality risk | Moderate | ~3.3% perplexity degradation | Higher (activations also quantized) |
| Compute throughput | Hardware FP4 TensorCores | Software emulation (slower) | Hardware FP4 TensorCores (SM100) |
| Production readiness | Available since v0.15.1 | Merged to main (April 22, 2026) | Merged to main (April 2026), pre-release |
| MoE support | FlashInfer backend | TritonExperts emulation | New CUTLASS kernel (PR #37463) |

## When to Use

- **NVFP4 W4A16 (Blackwell)**: large MoE models on Blackwell where FP8 still doesn't fit; validate quality per model
- **NVFP4 emulation (H100/MI300/MI350)**: run NVFP4-format checkpoints on non-Blackwell hardware; useful for portability when Blackwell is unavailable; accept ~3% quality penalty and slower compute vs native
- **MXFP4 W4A4**: only on SM100 (B200), for maximum compute throughput on MoE expert GEMMs; currently pre-release, requires quality validation
- **Avoid native FP4 on pre-Hopper**: FP4 emulation is now available for H100, but pre-H100 hardware has no viable FP4 path; use FP8 instead

## Open Questions

- What are the per-model quality deltas for W4A4 MXFP4 on major MoE models (DeepSeek V3/R1, Qwen-MoE, Mixtral)?
- How does the CUTLASS kernel's throughput compare to the FlashInfer MXFP4 MoE backend on B200?
- Will MXFP4 W4A4 be backported to SM120 or is it B200/GB200-exclusive?
- When does MXFP4 W4A4 ship in a numbered vLLM release?
- Does W4A4 interact with MRV2 (Model Runner V2) or does it require the V1 execution path?
- What is the throughput overhead of NVFP4 emulation (PR #35737) on H100 vs loading BF16 weights directly? The memory reduction may not be worth the emulation cost for latency-sensitive workloads.
- Does NVFP4 emulation extend to SM120 (DGX Spark, RTX PRO 6000) Blackwell consumer SKUs, or is it strictly for SM90/AMD?

## Sources

- [raw/2026-04-19-vllm-prs-apr17-19.md](../../raw/2026-04-19-vllm-prs-apr17-19.md) — PR #37463 CUTLASS W4A4 MoE kernel details
- [raw/vllm-benchmarks-2026.md](../../raw/vllm-benchmarks-2026.md) — vLLM Blackwell performance context
- [raw/2026-04-15-vllm-v019-release.md](../../raw/2026-04-15-vllm-v019-release.md) — v0.19.0 context for NVFP4 support status
- [raw/2026-04-23-vllm-prs-apr22-23.md](../../raw/2026-04-23-vllm-prs-apr22-23.md) — PR #35737 NVFP4 MoE emulation for H100/MI300/MI350
