---
title: "Speculative Decoding"
tags: [latency, throughput, decoding, speculation]
created: 2026-04-14
updated: 2026-04-14
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md]
related: [concepts/model-runner-v2.md, concepts/continuous-batching.md]
---

# Speculative Decoding

## Summary
Speculative decoding uses a small, fast "draft" model to predict multiple tokens ahead, then verifies them in a single forward pass of the large "target" model. Correct predictions are accepted for free; incorrect ones are rejected and regenerated. This can significantly reduce latency without changing output quality.

## Problem It Solves
Autoregressive decoding is inherently sequential — each token depends on the previous one. For large models, each forward pass has high latency even though the GPU is underutilized (especially for single-sequence decode). Speculative decoding amortizes the cost by verifying multiple tokens per forward pass.

## How It Works
1. **Draft phase**: a small model generates K candidate tokens quickly
2. **Verify phase**: the target model processes all K tokens in one forward pass
3. **Accept/reject**: using a rejection sampler, tokens are accepted if they match what the target model would have produced. The first rejected token is resampled from the target.
4. On average, multiple tokens are accepted per iteration, yielding a speedup proportional to the acceptance rate.

## Implementation in vLLM
- **EAGLE-3** supported (Q1 2026 roadmap: "Support and test EAGLE-3 thoroughly")
- **Zero-bubble async scheduling** — speculative decoding now works with async scheduling for overlap (PR #32951)
- **MRV2 integration** — rejection sampler with greedy/logprobs support (PRs #37238, #37237)
- **Multi-modal spec decode** — embeddings for vision models (PR #36097)

## Trade-offs
- Requires maintaining a second (draft) model in memory
- Acceptance rate varies by task — highly predictable text benefits most
- Adds complexity to the serving pipeline

## When to Use
- Latency-sensitive applications (interactive chat, coding assistants)
- When the draft model has a high acceptance rate for the target workload
- When GPU is underutilized during decode (low-concurrency scenarios)

## Open Questions
- What's the best draft model selection strategy for a given target model?
- How does EAGLE-3 compare to vanilla speculative decoding in practice?
- How does spec decode interact with chunked prefill scheduling?
