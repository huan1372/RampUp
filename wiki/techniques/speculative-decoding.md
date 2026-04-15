---
title: "Speculative Decoding"
tags: [latency, throughput, decoding, speculation]
created: 2026-04-14
updated: 2026-04-15
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md, raw/2026-04-15-p-eagle-blog.md, raw/2026-04-15-vllm-v019-release.md]
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
- **Zero-bubble async scheduling** — speculative decoding now works with async scheduling for overlap, default in v0.19.0 (PR #32951)
- **MRV2 integration** — rejection sampler with greedy/logprobs support (PRs #37238, #37237)
- **Multi-modal spec decode** — embeddings for vision models (PR #36097)
- **NGram GPU spec decode** (v0.18.0) — NGram drafting now runs on GPU, compatible with async scheduler

## P-EAGLE: Parallel Speculative Decoding
P-EAGLE (shipped in vLLM v0.16.0+) transforms EAGLE's autoregressive draft generation into a single parallel forward pass. (source: raw/2026-04-15-p-eagle-blog.md)

### How It Differs from EAGLE
Standard EAGLE generates K draft tokens sequentially (K forward passes of the draft model). P-EAGLE generates all K tokens in one pass using learnable "mask" parameters as placeholders for unknown future positions — those parameters are learned during drafter training.

### Performance vs. EAGLE-3

| Benchmark   | Concurrency | P-EAGLE speedup over EAGLE-3 |
|-------------|-------------|------------------------------|
| SPEED-Bench | 1           | **1.69×**                    |
| MT-Bench    | 1           | 1.55×                        |
| HumanEval   | 1           | 1.55×                        |
| MT-Bench    | 64          | 1.05× (diminishes at scale)  |
| HumanEval   | 64          | 1.23×                        |

**Acceptance length at K=7**: P-EAGLE shows ~30-31% higher acceptance rates than EAGLE-3 (HumanEval: 3.94 vs 3.03; SPEED-Bench: 3.38 vs 2.59)

**Optimal speculation depth**: P-EAGLE peaks at K=7; EAGLE-3 maxes at K=3. Larger K is economical because drafting costs a flat single forward pass regardless of K.

Pre-trained drafter models available for: GPT-OSS 120B, GPT-OSS 20B, Qwen3-Coder 30B.

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
- Why does P-EAGLE's advantage diminish at high concurrency (c=64)? Is this a batching overhead issue or acceptance rate regression?
- When will P-EAGLE drafter models be available for more target models beyond GPT-OSS and Qwen3-Coder?
