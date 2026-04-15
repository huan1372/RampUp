---
title: "Model Runner V2 (MRV2)"
tags: [vllm-core, architecture, execution]
created: 2026-04-14
updated: 2026-04-15
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md, raw/2026-04-15-model-runner-v2-blog.md, raw/2026-04-15-vllm-v019-release.md]
related: [concepts/paged-attention.md, techniques/speculative-decoding.md, techniques/tensor-parallelism.md]
---

# Model Runner V2 (MRV2)

## Summary
Model Runner V2 is a ground-up rewrite of vLLM's model execution layer, shipped in vLLM v0.18.0 (opt-in) and made the default in v0.19.0 (April 2026). It replaces V1 with a modular, GPU-native, async-first architecture. The largest source file shrank from 6,700+ lines (V1) to under 1,300 lines (MRV2). (source: raw/2026-04-15-model-runner-v2-blog.md)

## Core Design Principles
Three foundational changes drive MRV2:
1. **Modularity** — model-specific logic isolated in a `ModelState` abstract interface; common execution path stays generic
2. **GPU-native input preparation** — `input_ids`, `positions`, `query_start_loc`, `seq_lens` now built on GPU via Triton kernels; eliminates CPU-GPU data transfers for input construction
3. **Async-first** — target is zero CPU-GPU synchronization across all supported features; output transfers run in separate CUDA streams decoupled from computation

## Architecture Details

### Persistent Batch Redesign
V1 tightly coupled persistent state to per-step model inputs, requiring costly reordering. MRV2 maintains a stable state table independent of input layout and uses gather operations to produce correctly ordered inputs each step. (source: raw/2026-04-15-model-runner-v2-blog.md)

### Sampler Improvements
- **Gumbel-Max kernel** — avoids softmax materialization using stateless in-kernel RNG
- **Top-k logprobs** — computes logprobs only for selected top-k candidates, not all vocab
- **Prompt logprobs** — fine-grained chunking including within a single prompt
- **Indirection support** (`idx_mapping`) inside kernels reduces state expansion

### Feature Coverage (v0.19.0)
- Piecewise CUDA graphs for pipeline parallelism (PR #35162)
- ViT full CUDA graph capture (PR #35963)
- Spec decode rejection sampler with greedy/logprobs (PRs #37238, #37237)
- Multi-modal embeddings for spec decode (PR #36097)
- EPLB (Expert-Parallel Load Balancing) (PR #37488)
- Streaming inputs (PR #37028)

## Performance Data
- **Throughput**: Qwen3-0.6B on GB200 shows **56% throughput increase** from GPU-native input prep alone (25K vs. 16K tokens/s) (source: raw/2026-04-15-model-runner-v2-blog.md)
- **Spec decode**: **6.3% lower TPOT** on 4×GB200 with GLM-4.7-FP8 + MTP=1, from eliminating CPU-GPU sync points
- **Overall**: 1.7× throughput over V0 engine when combined with async scheduling (source: raw/2026-04-15-vllm-v019-release.md)

## Current Status
MRV2 is the **default** execution path as of v0.19.0. V1 remains for unsupported cases. Known V2 gaps (as of v0.18.0): linear attention models (Qwen3.5, Nemotron 3 Super), non-Eagle/MTP spec decode methods, EPLB, DBO, logits processors, LoRA — being closed in Q2 2026.

Enable on v0.18.x: `export VLLM_USE_V2_MODEL_RUNNER=1`. No API changes.

## Relationship to Other Concepts
- Executes models using [PagedAttention](paged-attention.md) for memory management
- Integrates with [Speculative Decoding](../techniques/speculative-decoding.md) via the rejection sampler
- Supports [Tensor Parallelism](../techniques/tensor-parallelism.md) via piecewise CUDA graphs

## Open Questions
- Which models still require MRV1 as of v0.19? When will they migrate?
- What's the cold-start overhead of CUDA graph capture in MRV2 vs MRV1?
- When does EPLB support land in MRV2?
