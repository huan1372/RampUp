---
title: "Model Runner V2 (MRV2)"
tags: [vllm-core, architecture, execution]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md]
related: [concepts/paged-attention.md, techniques/speculative-decoding.md, techniques/tensor-parallelism.md]
---

# Model Runner V2 (MRV2)

## Summary
Model Runner V2 is a ground-up rewrite of vLLM's model execution layer, released March 2026. It replaces the original Model Runner V1 with a cleaner, more modular architecture that better supports CUDA graphs, speculative decoding, and multi-modal models.

## How It Works
The model runner is the component that actually executes forward passes through the model. MRV2 redesigns this with:

- **Modular architecture** — cleaner separation of concerns, making it easier to add new features
- **Piecewise CUDA graphs** — supports CUDA graph capture for pipeline parallelism (PR #35162)
- **ViT full CUDA graphs** — vision encoders now support full CUDA graph capture (PR #35963)
- **Spec decode integration** — rejection sampler with greedy/logprobs support (PRs #37238, #37237)
- **Multi-modal embeddings** — for speculative decoding (PR #36097)
- **Streaming inputs** — support for streaming input data (PR #37028)
- **EPLB support** — Expert-Parallel Load Balancing (PR #37488)

## Current Status
MRV2 is the default execution path in vLLM v0.19.0, but MRV1 remains for "long tail" use cases. The Q2 2026 roadmap focuses on:
- Continuing to fill gaps in MRV2 coverage
- Moving more optimizations to MRV2 (Inductor partition, attn+quant fusion, Async TP)
- Eventually deprecating MRV1

## Relationship to Other Concepts
- Executes models using [PagedAttention](paged-attention.md) for memory management
- Integrates with [Speculative Decoding](../techniques/speculative-decoding.md) via the rejection sampler
- Supports [Tensor Parallelism](../techniques/tensor-parallelism.md) via piecewise CUDA graphs

## Open Questions
- Which models still require MRV1? What's the migration timeline?
- What's the cold-start overhead of CUDA graph capture in MRV2 vs MRV1?
- How does the modular design affect extensibility for custom model architectures?
