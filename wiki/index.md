---
title: "Master Index"
tags: [index, meta]
created: 2026-04-14
updated: 2026-04-24
---

# Inference Optimization Knowledge Base — Index

## Overview
- [Overview & Synthesis](overview.md)

## Concepts
- [PagedAttention](concepts/paged-attention.md) — virtual memory for KV cache
- [Continuous Batching](concepts/continuous-batching.md) — dynamic request scheduling
- [Chunked Prefill](concepts/chunked-prefill.md) — splitting long-context prefills
- [KV Cache Management](concepts/kv-cache-management.md) — memory allocation strategies
- [Model Runner V2](concepts/model-runner-v2.md) — vLLM's new execution core; warm compile time −88–96% via FX graph inlining (PR #40151, Apr 23 2026); RayExecutorV2 + AOT batch-invariance in v0.20.0
- [DeepSeek V4 Hybrid Attention](concepts/deepseek-v4-attention.md) — CSA + HCA + mHC; 10% of V3.2's KV cache at 1M context; vLLM day-0 support (Apr 24 2026)

## Techniques
- [FP8 Quantization](techniques/fp8-quantization.md) — near-free 2x memory reduction; MLA+Group FP8 fusion for DeepSeek (PR #38877, Apr 22 2026)
- [FP4 / MXFP4 Quantization](techniques/fp4-quantization.md) — ~4× memory reduction on Blackwell; CUTLASS W4A4 MoE kernel (merged Apr 2026); NVFP4 emulation fallback for H100/MI300/MI350 (PR #35737, Apr 22 2026); MXFP8 Marlin GEMM in v0.20.0
- [KV Cache Quantization](techniques/kv-cache-quantization.md) — sub-FP8 compression spectrum; TurboQuant 2-bit (4× capacity, v0.20.0), 2.6–4.9× with FA3/FA4 prefill support; per-token-head INT8/FP8 (v0.20.0); IsoQuant SO(4) rotation family (research); sequential compression beyond Shannon limit (research)
- [Cross-Layer KV Compression](techniques/cross-layer-kv-compression.md) — architectural 50% KV reduction; YOCO++ (Apr 2026 research, not yet in vLLM)
- [Speculative Decoding](techniques/speculative-decoding.md) — draft-verify acceleration; P-EAGLE, CSD (2.33× speedup); SpecGuard; CPU draft-model (v0.20.0); CUDA graph Eagle prefill (v0.20.0); PR #40654 sync elimination
- [Prefix Caching](techniques/prefix-caching.md) — reusing KV cache across requests
- [Disaggregated Serving](techniques/disaggregated-serving.md) — separating prefill and decode; StreamServe adaptive spec depth; Prefill-as-a-Service cross-DC; 3FS KVConnector (v0.20.0); EPLB improved comms (v0.20.0); Nixl 0.10.1
- [Tensor Parallelism](techniques/tensor-parallelism.md) — splitting models across GPUs

## Changelog
- [Week of 2026-04-14](changelog/week-2026-04-14.md) — initial knowledge base seeding
- [Collect run 2026-04-15](changelog/collect-2026-04-15.md) — vLLM v0.18/v0.19, MRV2 deep-dive, P-EAGLE, async KV prefetch paper
- [Ingest run 2026-04-15 (ramp-up recap)](changelog/ingest-2026-04-15-rampup-recap.md) — PDF Q&A session ingested; statistical multiplexing, continuous-batching timeline, glossary
- [Collect run 2026-04-16](changelog/collect-2026-04-16.md) — TurboQuant KV cache compression (PR #38479, merged Apr 15); new kv-cache-quantization page
- [Collect run 2026-04-19](changelog/collect-2026-04-19.md) — MXFP4 W4A4 CUTLASS MoE kernel (PR #37463); CSD speculative decoding (arXiv 2604.13634); new fp4-quantization page
- [Collect run 2026-04-20](changelog/collect-2026-04-20.md) — SpecGuard step-level verification (arXiv 2604.15244); StreamServe disaggregated+adaptive spec decode (arXiv 2604.09562); Prefill-as-a-Service cross-datacenter serving (arXiv 2604.15039)
- [Collect run 2026-04-21](changelog/collect-2026-04-21.md) — vLLM v0.19.1 release; YOCO++ cross-layer KV (arXiv 2604.13556); FP16 KV divergence finding (arXiv 2604.15409)
- [Collect run 2026-04-22](changelog/collect-2026-04-22.md) — vLLM PRs Apr 21–22 (MLA+FP8 fusion, CUDAGraph profiling default, fused RMS norm, KV offload multi-group); IsoQuant SO(4) rotation (arXiv 2603.28430); sequential KV compression (arXiv 2604.15356)
- [Collect run 2026-04-23](changelog/collect-2026-04-23.md) — vLLM PRs Apr 22–23 (TurboQuant FA3/FA4 prefill +71–89% throughput on Hopper; NVFP4 MoE emulation for H100/MI300/MI350; FX graph inlining −88–96% warm compile time)
- [Collect run 2026-04-24](changelog/collect-2026-04-24.md) — vLLM v0.20.0 major release (PyTorch 2.11+CUDA 13.0, FA4 default MLA, TurboQuant 2-bit, online quant frontend, 3FS KVConnector); DeepSeek V4 day-0 support (CSA+HCA+mHC, 10% V3.2 KV at 1M ctx); new deepseek-v4-attention concept page
