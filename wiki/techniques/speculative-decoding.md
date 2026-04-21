---
title: "Speculative Decoding"
tags: [latency, throughput, decoding, speculation]
created: 2026-04-14
updated: 2026-04-21
sources: [raw/vllm-releases.md, raw/vllm-roadmap-q2-2026.md, raw/2026-04-15-p-eagle-blog.md, raw/2026-04-15-vllm-v019-release.md, raw/2026-04-19-calibrated-speculative-decoding-arxiv.md, raw/2026-04-20-specguard-arxiv-2604-15244.md, raw/2026-04-20-streamserve-arxiv-2604-09562.md, raw/2026-04-21-vllm-v0191-release.md]
related: [concepts/model-runner-v2.md, concepts/continuous-batching.md, techniques/disaggregated-serving.md]
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

## Eagle3 + Gemma4 (vLLM v0.19.1)

vLLM v0.19.1 (April 18, 2026) added Eagle3 speculative decoding support for Gemma4 models via PR #39450. This is the first Eagle-family drafter for a Google-origin MoE+multimodal architecture. Key context:
- Gemma4 is a hybrid dense+MoE model with multimodal and reasoning capabilities
- The Eagle3 drafter runs alongside the Gemma4 target model, requiring the additional drafter to fit in GPU memory
- Requires `transformers>=5.5.3` (resolved by the v0.19.1 dependency upgrade)
- Quantized MoE inference for Gemma4 (PR #39045) is available in the same release, allowing the Gemma4 target model to run in FP8/quantized mode while Eagle3 provides speculative drafts

(source: raw/2026-04-21-vllm-v0191-release.md)

## Trade-offs
- Requires maintaining a second (draft) model in memory
- Acceptance rate varies by task — highly predictable text benefits most
- Adds complexity to the serving pipeline

## When to Use
- Latency-sensitive applications (interactive chat, coding assistants)
- When the draft model has a high acceptance rate for the target workload
- When GPU is underutilized during decode (low-concurrency scenarios)

## Calibrated Speculative Decoding (CSD) — Research (arXiv 2604.13634, April 2026)

CSD is a training-free framework that addresses **false rejections** — cases where the draft model's token is semantically correct but lexically diverges from the target model's top token, causing the standard rejection sampler to discard a valid output.

**Two modules**:
1. **Online Correction Memory (OCM)**: accumulates rejected (draft token, context) pairs at runtime; identifies recurring divergence patterns and proposes "rescue candidates" drawn from historical rejections
2. **Semantic Consistency Gating (SCG)**: validates rescue candidates via a probability ratio test (draft / target), admitting only tokens within a calibrated threshold — prevents hallucinated rescues

**Results**: Peak throughput speedup of **2.33×** over baseline speculative decoding across diverse LLMs (exact models not specified). Training-free — plugs into any existing draft-verify pipeline.

**vLLM integration status**: Not yet integrated. Authors explicitly note vLLM integration as needed for production deployment. Implementation requires changes to vLLM's rejection sampler (`vllm/model_executor/layers/spec_decode/`) and addition of a per-worker or global correction memory store.

(source: raw/2026-04-19-calibrated-speculative-decoding-arxiv.md)

## SpecGuard: Step-Level Verification for Reasoning (arXiv 2604.15244, April 2026)

Standard spec decode is token-centric: a rejected token triggers rollback, but an accepted erroneous reasoning step propagates forward corrupting the chain. SpecGuard adds step-level verification using model-internal signals only.

**Mechanism**:
1. Draft model samples multiple candidate steps (not just one)
2. Self-consistency scoring selects the most consistent candidate
3. An ensemble of lightweight verifiers (distilled from target model hidden states) validates semantic soundness
4. Accept or recompute the step from target; only the step is recomputed, not the full sequence

**Results** (multiple reasoning benchmarks, vs standard SD and reward-guided SD):

| Metric | vs Standard SD | vs Reward-guided SD |
|--------|---------------|---------------------|
| Accuracy | +3.6% | better |
| Latency | −11% | better |

No external reward model — overhead is proportional to draft model, not target.

**vLLM integration**: Not yet implemented. Requires step-boundary detection + multi-candidate draft sampling in vLLM's spec decode framework.

**Comparison with other approaches**: SpecGuard targets reasoning correctness; [CSD](speculative-decoding.md#calibrated-speculative-decoding-csd--research-arxiv-260413634-april-2026) targets false-rejection throughput; [P-EAGLE](speculative-decoding.md#p-eagle-parallel-speculative-decoding) targets draft-generation speed. Complementary, not competing.

(source: raw/2026-04-20-specguard-arxiv-2604-15244.md)

## SpecuStream: Runtime-Adaptive Speculation Depth (StreamServe, arXiv 2604.09562)

A component of the [StreamServe](disaggregated-serving.md#streamserve-adaptive-speculative-flows-arxiv-260409562-april-2026) disaggregated serving system. SpecuStream adjusts speculation depth K online from live acceptance rate signals.

**Problem**: Static K is suboptimal for reasoning tasks — during early problem setup, acceptance rates are high (predictable boilerplate); during mid-problem derivation steps, entropy spikes and acceptance drops, wasting draft computation at the original K.

**Mechanism**: Per-stream running average of accepted tokens per verification step. K increases when rate exceeds threshold; K decreases when rate falls below floor. Adjustments happen per request batch without stopping inference.

**Effect**: Combined with disaggregated serving (PipeServe-Engine), StreamServe achieves 9.5% throughput improvement on GSM8K (264 tok/s vs 241 tok/s TP vLLM) and 11–18× latency reduction. See [Disaggregated Serving](disaggregated-serving.md) for full benchmark table.

(source: raw/2026-04-20-streamserve-arxiv-2604-09562.md)

## Open Questions
- What's the best draft model selection strategy for a given target model?
- How does EAGLE-3 compare to vanilla speculative decoding in practice?
- How does spec decode interact with chunked prefill scheduling?
- Why does P-EAGLE's advantage diminish at high concurrency (c=64)? Is this a batching overhead issue or acceptance rate regression?
- When will P-EAGLE drafter models be available for more target models beyond GPT-OSS and Qwen3-Coder?
- Does CSD's 2.33× peak speedup hold at high concurrency, or does it share P-EAGLE's diminishing returns pattern?
- What is the memory overhead of OCM's correction history at serving scale (thousands of concurrent requests)?
- SpecGuard uses "lightweight verifiers distilled from target hidden states" — what is the distillation cost and when must it be redone for a new target model?
- Does SpecuStream's adaptive K work well with P-EAGLE's single-pass drafting (where K is a training-time parameter, not runtime-tunable)?
