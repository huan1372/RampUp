---
title: "Master Index"
tags: [index, meta]
created: 2026-04-14
updated: 2026-04-15
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
- [Speculative Decoding](techniques/speculative-decoding.md) — draft-verify acceleration
- [Prefix Caching](techniques/prefix-caching.md) — reusing KV cache across requests
- [Disaggregated Serving](techniques/disaggregated-serving.md) — separating prefill and decode
- [Tensor Parallelism](techniques/tensor-parallelism.md) — splitting models across GPUs

## Changelog
- [Week of 2026-04-14](changelog/week-2026-04-14.md) — initial knowledge base seeding
- [Collect run 2026-04-15](changelog/collect-2026-04-15.md) — vLLM v0.18/v0.19, MRV2 deep-dive, P-EAGLE, async KV prefetch paper
- [Ingest run 2026-04-15 (ramp-up recap)](changelog/ingest-2026-04-15-rampup-recap.md) — PDF Q&A session ingested; statistical multiplexing, continuous-batching timeline, glossary
