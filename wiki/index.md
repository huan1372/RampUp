---
title: "Master Index"
tags: [index, meta]
created: 2026-04-14
updated: 2026-04-28
---


# Inference Optimization Knowledge Base — Index

## Overview
- [Overview & Synthesis](overview.md)

## Concepts
- [PagedAttention](concepts/paged-attention.md) — virtual memory for KV cache; Ragged Paged Attention for TPU: 86% MBU decode, 73% MFU prefill, 5× throughput (arXiv 2604.15464, Apr 2026)
- [Continuous Batching](concepts/continuous-batching.md) — dynamic request scheduling
- [Chunked Prefill](concepts/chunked-prefill.md) — splitting long-context prefills; StepPool embedding alignment fix for 4K+ token sequences (PR #41049, Apr 28 2026)
- [KV Cache Management](concepts/kv-cache-management.md) — memory allocation strategies; TTKV temporal tiering (76% latency reduction on 128K ctx, arXiv 2604.19769); HybridGen CPU-GPU hybrid attention (1.41×–3.2× throughput, arXiv 2604.18529); HMA multi-group KV offload store (PR #39403, Apr 25 2026); SWA/chunked-local scheduler admission deadlock fix (PR #40946, Apr 27 2026)
- [Model Runner V2](concepts/model-runner-v2.md) — vLLM's new execution core; warm compile time −88–96% via FX graph inlining (PR #40151, Apr 23 2026); RayExecutorV2 + AOT batch-invariance in v0.20.0; Eagle prefill metadata skip ~5-10% latency improvement (PR #40410, Apr 27 2026)
- [DeepSeek V4 Hybrid Attention](concepts/deepseek-v4-attention.md) — CSA + HCA + mHC; 10% of V3.2's KV cache at 1M context; vLLM day-0 support (Apr 24 2026); DSML streaming fix (PR #40806, Apr 26 2026); SiLU clamp for shared expert (PR #40950, Apr 27 2026)

## Techniques
- [FP8 Quantization](techniques/fp8-quantization.md) — near-free 2x memory reduction; MLA+Group FP8 fusion for DeepSeek (PR #38877, Apr 22 2026); Humming JIT kernel W{1–8}A{16/8/4} (PR #34556, Apr 24 2026, ~1.58× over Marlin on H20); FP8 FlashInfer ViT attention for Qwen3 VL (PR #38065, Apr 27 2026, 1.18× at 4K/GB200)
- [FP4 / MXFP4 Quantization](techniques/fp4-quantization.md) — ~4× memory reduction on Blackwell; CUTLASS W4A4 MoE kernel (merged Apr 2026); NVFP4 emulation fallback for H100/MI300/MI350 (PR #35737, Apr 22 2026); MXFP8 Marlin GEMM in v0.20.0
- [KV Cache Quantization](techniques/kv-cache-quantization.md) — sub-FP8 compression spectrum; TurboQuant 2-bit (4× capacity, v0.20.0), 2.6–4.9× with FA3/FA4 prefill support; TurboQuant shared dequant buffers (57.6 GB saved at 1M ctx, CUDA Graph enabled, PR #40941, Apr 27 2026); GRACE graph-guided channel elimination (60% KV reduction, arXiv 2604.16983); IsoQuant SO(4) rotation family (research); sequential compression beyond Shannon limit (research)
- [Cross-Layer KV Compression](techniques/cross-layer-kv-compression.md) — architectural 50% KV reduction; YOCO++ (Apr 2026 research, not yet in vLLM)
- [CPU-GPU Hybrid Attention](techniques/cpu-gpu-hybrid-attention.md) — HybridGen: CPU as active attention compute participant via CXL memory; 1.41×–3.2× over baselines (arXiv 2604.18529, Apr 2026)
- [Speculative Decoding](techniques/speculative-decoding.md) — draft-verify acceleration; P-EAGLE, CSD (2.33× speedup); SMC-SD particle resampling (2.36× over standard SD, arXiv 2604.15672); SpecGuard; CPU draft-model (v0.20.0); CUDA graph Eagle prefill (v0.20.0); DiP-SD distributed pipelined SD at edge (arXiv 2604.20919, Apr 2026); Eagle prefill metadata skip ~5-10% latency (PR #40410, Apr 27 2026); independent drafter attention backend (PR #39930, Apr 28 2026)
- [Prefix Caching](techniques/prefix-caching.md) — reusing KV cache across requests
- [Disaggregated Serving](techniques/disaggregated-serving.md) — separating prefill and decode; StreamServe adaptive spec depth; Prefill-as-a-Service cross-DC; 3FS KVConnector (v0.20.0); EPLB improved comms (v0.20.0); Nixl 0.10.1; NIXL EP batched-expert fix (PR #40412, Apr 24 2026); FlashInfer NVLink MNNVL workspace sizing fix (PR #40893, Apr 26 2026); full KV offload for remote decode in P/D (PR #40346, Apr 26 2026)
- [Tensor Parallelism](techniques/tensor-parallelism.md) — splitting models across GPUs; ReaLB multimodal MoE EP load balancing via dynamic precision (1.29× layer speedup, implemented in vLLM, arXiv 2604.19503); EPLB replica bias fix via Knuth hash (1.2→1.07 max/mean ratio, PR #40810, Apr 24 2026); MoE routed output unpad fix for GPT-OSS/B200 (PR #40865, Apr 25 2026)

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
- [Collect run 2026-04-24](changelog/collect-2026-04-24.md) — vLLM v0.20.0 major release (PyTorch 2.11+CUDA 13.0, FA4 default MLA, TurboQuant 2-bit, online quant frontend, 3FS KVConnector); DeepSeek V4 day-0 support (CSA+HCA+mHC, 10% V3.2 KV at 1M ctx); new deepseek-v4-attention concept page; deferred papers: TTKV, HybridGen, SMC-SD, GRACE, ReaLB, Ragged Paged Attention (TPU); new cpu-gpu-hybrid-attention technique page
- [Collect run 2026-04-25](changelog/collect-2026-04-25.md) — vLLM post-v0.20.0 PRs: Humming JIT quantization kernel (W{1–8}A{16/8/4}, ~1.58× over Marlin on H20); EPLB replica selection bias fix (Knuth hash, 1.07 max/mean); NIXL EP batched-expert consistency fix; ROCm memory leak fix; no new arXiv papers passing relevance filter
- [Collect run 2026-04-26](changelog/collect-2026-04-26.md) — vLLM post-v0.20.0 PRs: FlashInfer NVLink MNNVL workspace sizing fix (DP/EP mismatch crash, PR #40893); MoE routed output unpad fix (GPT-OSS B200, PR #40865); HMA multi-group KV offload store (PR #39403); DSML streaming fix for DeepSeek V4/3.2 (PR #40806); DiP-SD edge distributed pipelined speculative decoding (arXiv 2604.20919)
- [Collect run 2026-04-27](changelog/collect-2026-04-27.md) — vLLM post-v0.20.0 PRs: TurboQuant shared dequant buffers (57.6 GB saved at 1M ctx, CUDA Graph enabled, PR #40941); DeepSeek V4 SiLU clamp for shared expert (PR #40950); FP8 FlashInfer ViT attention for Qwen3 VL (1.18× at 4K/GB200, PR #38065); SWA scheduler admission deadlock fix (PR #40946); full KV offload for remote decode (PR #40346)
- [Collect run 2026-04-28](changelog/collect-2026-04-28.md) — vLLM post-v0.20.0 PRs: Eagle prefill metadata skip (PR #40410, ~5-10% latency); independent drafter attention backend selection (PR #39930); StepPool embedding alignment fix for chunked prefill (PR #41049); TRTLLM MoE routing update with SigmoidRenorm+MiniMax2 (PR #39141)
