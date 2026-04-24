---
title: "vLLM v0.20.0 Release Notes"
source_type: github-release
source_url: https://github.com/vllm-project/vllm/releases/tag/v0.20.0
collected: 2026-04-24
tags: [vllm, release, cuda13, pytorch2.11, fa4, turboquant, quantization, eplb, moe, disaggregated-serving]
---

# vLLM v0.20.0 Release Notes

**Released:** April 23, 2026  
**Commits:** 546 from 257 contributors (83 new)  
**Source:** https://github.com/vllm-project/vllm/releases/tag/v0.20.0

## Core Dependencies (Breaking Changes)

- **PyTorch 2.11 + CUDA 13.0 as defaults** — CUDA wheel now ships CUDA 13.0; PyTorch bumped to torch 2.11 for CUDA environments. XPU remains torch-xpu 2.10. This is a hard dependency bump; environments on CUDA 12.x must update.
- **Transformers v5 compatibility** — achieves full v4/v5 compat; ongoing fixes for v4/v5 API surface divergence.
- **FlashAttention 4** — upstream sync; FA4 installs via symlink-on-install behavior.
- **DeepGEMM integrated into vLLM wheel** — via CMake; no separate install step needed.
- **Nixl bumped to 0.10.1** — KV transfer infrastructure update.

## Attention & Performance

### FA4 as Default MLA Prefill Backend
FlashAttention 4 re-enabled as the default MLA prefill backend with:
- Head-dim 512 support
- Paged-KV support on SM90+ GPUs (H100/H200/H20 and Blackwell)
This corrects the previous regression where FA2 was defaulted on Hopper/Blackwell MLA paths.

### TurboQuant 2-bit KV Cache
New attention backend delivering **2-bit KV cache compression with 4× capacity** increase. This extends the TurboQuant family below the 3-bit floor available in prior releases. Now ships in a numbered release for the first time (TurboQuant 4/3-bit merged April 15 as PR #38479; 2-bit added for v0.20.0).

### Per-Token-Head INT8/FP8 KV Cache Quantization
New fine-grained quantization granularity: per-token-head (rather than per-layer or per-tensor). Allows more precise scale factors, reducing quantization error for models with high KV activation variance across heads.

### Online Quantization Frontend
New end-to-end online quantization interface. Allows applying quantization (INT8, FP8, NVFP4) to model weights at load time without pre-quantized checkpoints. Includes documentation.

## vLLM IR (Intermediate Representation)
Initial skeleton introduced with `rms_norm` operation. Foundation for future Helion-based custom kernel generation. Not yet user-facing; enables internal kernel development infrastructure.

## Engine & Scheduling

### RayExecutorV2
New distributed execution backend using Ray. Replaces or extends the previous Ray-based executor path. Enables cleaner distributed state management.

### EPLB (Expert Parallel Load Balancing) Improvements
- Alternative communication strategy for weight exchange during expert load balancing
- Router record for prefill mapping optimization
Existing EPLB support (first added PR #37488 in v0.19.0) now has enhanced communication efficiency.

### KV Offload & Connector
- **3FS KVConnector introduced** — new KV store connector for Bytedance 3FS distributed file system, enabling KV cache sharing across nodes via 3FS
- **Unified memory layout for offloading workers** — simplifies multi-backend offloading
- **cache_salt propagation** — per-user KV cache isolation via MP connector
- **LMCache optimizations** — block-allocation event handling, MP save optimization with MLA

### Disaggregated Serving (Nixl / NIXL)
- Nixl bumped to 0.10.1 (from 0.9.x)
- Heterogeneous TP 3-read conv-state transfer for NIXL + Mamba

### Scheduling & MM Overhead
- `mm-scheduler get_num_embed` overhead reduced (multimodal scheduler)
- Labeled waiting-breakdown metrics (capacity/deferred) added
- Safe request abort when FSM fails to advance

## Compilation (torch.compile)
- **Opaque Objects support** on torch 2.11
- **AOT compile with batch-invariance mode** — allows compile artifacts to be shared across different batch sizes
- **Inductor cache nested under AOT directory**
- **FX graph splitting via codegen**
- **Inductor pre-grad passes re-enabled** for torch≥2.12

## Hardware-Specific Optimizations

### NVIDIA (CUDA)
- swapAB for SM120 CUTLASS blockwise FP8 GEMM (Blackwell consumer/pro SKUs)
- MXFP4 W4A4 CUTLASS MoE for SM100 (B200)
- TRTLLM GEN NVFP4 MoE with non-512-aligned hidden dims via weight padding
- **Fused qknorm+rope kernel on SM9.0** — fuses QK layer norm + RoPE into single kernel on H100
- Batched KV-cache swap via cuMemcpyBatchAsync
- ViT full CUDA graph for Qwen3-VL video processing
- Fused FP8 output quantization into merge_attn_states
- Full CUDA graph support for FlexAttention and Eagle prefill
- SiLU block-quant fusion v1

### AMD ROCm
- ZenCPU/AMD Zen CPU backend via zentorch
- RDNA 3.5/4 device IDs (gfx1150/1151/1201)
- MORI EP for unquantized MoE with AITER
- TritonW4A16LinearKernel implementation
- Asymmetric INT8 in TritonInt8ScaledMMLinearKernel
- NVFP4 dense models on MI300/MI355X via emulation (same Triton backend as H100 emulation)

### CPU Backend
- **CPU draft-model speculative decoding enabled** — first time CPU-hosted draft models work in vLLM speculative pipeline
- CPU int8 compute mode in AWQ
- Head_size 512 in cpu_attn
- W4A16 Autoround on CPU
- IBM Z s390x torch 2.11 builds

### DeepSeek / MLA
- Persistent TopK scheduler for MoE
- Fused weights projection
- Triton optimization for MLA decode

## Quantization & Compression Summary

| Format | Status in v0.20.0 | Hardware |
|--------|-------------------|----------|
| FP8 KV | Stable | H100/H200/B200 |
| TurboQuant 2-bit KV | New in v0.20.0 | SM90+ (Hopper, Blackwell) |
| TurboQuant 3/4-bit KV | Merged Apr 15 main; in v0.20.0 | SM90+ |
| Per-token-head INT8/FP8 KV | New in v0.20.0 | All CUDA |
| NVFP4 MoE (Blackwell native) | Existing | SM100 (B200) |
| NVFP4 dense (H100 emulation) | From PR #35737 | SM90 emulated |
| NVFP4 dense (MI300/MI355X emulation) | New in v0.20.0 | AMD |
| MXFP4 W4A4 CUTLASS MoE | Merged Apr 2026; in v0.20.0 | SM100 (B200) |
| MXFP8 Marlin GEMM | New in v0.20.0 | SM90+ |
| W8A8 MXFP8 linear/MoE | New in v0.20.0 | SM90+ |

## Performance Numbers Mentioned
- Mean-pooling optimization: ~5.9% throughput improvement
- Redundant-sync removal for pooling models: ~3.7% throughput improvement
- TurboQuant 2-bit: 4× KV capacity increase (vs BF16 baseline)
- Fused qknorm+rope on SM9.0: performance improvement (no specific number given)

## Breaking Changes Summary
1. PyTorch 2.11 + CUDA 13.0 required for CUDA wheels
2. `vllm:prompt_tokens_recomputed` metric removed; replaced with PrefillStats (`num_cached_tokens`, `num_external_computed_tokens`)
3. PoolerConfig rename: `logit_bias`/`logit_scale` → `logit_mean`/`logit_sigma`
4. Async scheduling disabled by default for pooling models
5. Petit NVFP4 quantization removed
6. V0 deprecations: `cprofile`/`cprofile_context`, accept output buffer in attention

## Security Fix
- SSRF fix in batch runner `download_bytes_from_url` function

## Dependency Changes
- fastsafetensors added to NVIDIA Dockerfile
- Helion bumped 0.3.2 → 0.3.3
- resampy dependency dropped
- librosa direct dependency dropped
- pyav and soundfile moved to common requirements
