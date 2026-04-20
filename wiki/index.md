---
title: "Master Index"
tags: [index, meta]
created: 2026-04-14
updated: 2026-04-20
---

# Inference Optimization Knowledge Base — Index

## Overview
- [Overview & Synthesis](overview.md)

## Concepts
- [PagedAttention](concepts/paged-attention.md) — virtual memory for KV cache
- [Continuous Batching](concepts/continuous-batching.md) — dynamic request scheduling
- [Chunked Prefill](concepts/chunked-prefill.md) — splitting long-context prefills
- [KV Cache Management](concepts/kv-cache-management.md) — memory allocation strategies
- [Model Runner V2](concepts/model-runner-v2.md) — vLLM's new execution core

## Techniques
- [FP8 Quantization](techniques/fp8-quantization.md) — near-free 2x memory reduction
- [FP4 / MXFP4 Quantization](techniques/fp4-quantization.md) — ~4× memory reduction on Blackwell; CUTLASS W4A4 MoE kernel (merged Apr 2026)
- [KV Cache Quantization](techniques/kv-cache-quantization.md) — sub-FP8 compression spectrum; TurboQuant 2.6–4.9× (merged Apr 2026); WHT overhead reduced (Apr 2026)
- [Speculative Decoding](techniques/speculative-decoding.md) — draft-verify acceleration; P-EAGLE, CSD (2.33× speedup); SpecGuard step-level verification (+3.6% acc, −11% lat); SpecuStream adaptive K (Apr 2026)
- [Prefix Caching](techniques/prefix-caching.md) — reusing KV cache across requests
- [Disaggregated Serving](techniques/disaggregated-serving.md) — separating prefill and decode; StreamServe adaptive spec depth (Apr 2026); Prefill-as-a-Service cross-DC via hybrid-attention (Apr 2026)
- [Tensor Parallelism](techniques/tensor-parallelism.md) — splitting models across GPUs

## Changelog
- [Week of 2026-04-14](changelog/week-2026-04-14.md) — initial knowledge base seeding
- [Collect run 2026-04-15](changelog/collect-2026-04-15.md) — vLLM v0.18/v0.19, MRV2 deep-dive, P-EAGLE, async KV prefetch paper
- [Ingest run 2026-04-15 (ramp-up recap)](changelog/ingest-2026-04-15-rampup-recap.md) — PDF Q&A session ingested; statistical multiplexing, continuous-batching timeline, glossary
- [Collect run 2026-04-16](changelog/collect-2026-04-16.md) — TurboQuant KV cache compression (PR #38479, merged Apr 15); new kv-cache-quantization page
- [Collect run 2026-04-19](changelog/collect-2026-04-19.md) — MXFP4 W4A4 CUTLASS MoE kernel (PR #37463); CSD speculative decoding (arXiv 2604.13634); new fp4-quantization page
- [Collect run 2026-04-20](changelog/collect-2026-04-20.md) — SpecGuard step-level verification (arXiv 2604.15244); StreamServe disaggregated+adaptive spec decode (arXiv 2604.09562); Prefill-as-a-Service cross-datacenter serving (arXiv 2604.15039)
