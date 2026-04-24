---
title: "Faster LLM Inference via Sequential Monte Carlo (SMC-SD)"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.15672
collected: 2026-04-24
tags: [speculative-decoding, throughput, latency, monte-carlo, sampling, inference]
---

# Faster LLM Inference via Sequential Monte Carlo

**arXiv:** 2604.15672  
**Submitted:** April 17, 2026  
**Authors:** Yahya Emara et al. — Cornell University, MIT, ETH Zürich

## Abstract / Core Claim

Standard speculative decoding (SD) uses rejection sampling: when the target model disagrees with the draft model on a token, the draft block is truncated at the first rejection. This truncation is the primary throughput bottleneck when draft and target distributions diverge. SMC-SD replaces token-level rejection with **importance-weighted resampling over a population of draft particles**, eliminating rollback and enabling fixed-size parallel verification.

## Core Algorithm

### Standard Speculative Decoding (baseline)
1. Draft model generates K candidate tokens autoregressively
2. Target model verifies all K in one parallel forward pass
3. Rejection sampler accepts prefix of tokens until first mismatch
4. First rejected token resampled from target; remainder discarded
5. Effective tokens accepted per step = E[acceptance length] < K

### SMC-SD
1. Draft model generates N particles (N independent K-token drafts)
2. Target model scores all particles in parallel — importance weights assigned based on likelihood ratio (target/draft)
3. Resampling: select the next token position using weights (no rejection, no rollback)
4. Verification becomes a fixed-size vectorized operation
5. Effective tokens accepted per step is higher than rejection sampling at similar draft-target divergence

### Key Properties
- **No rollback**: importance weighting replaces the truncation step
- **Approximate but bounded**: SMC-SD is a principled approximate inference scheme with per-step approximation error bounds
- **Uses idle compute**: LLM inference is memory bandwidth-bound; the arithmetic for particle scoring comes nearly for free during the memory-bound attention computation
- **Fixed-size verification**: removes the variable-length dependency of rejection sampling

## Performance Results

| Metric | vs Autoregressive | vs Standard Spec Decode |
|--------|------------------|------------------------|
| Speedup | **5.2×** | **2.36×** |
| Accuracy | within 3% of target model | within 3% of target model |

Benchmarks: reasoning, instruction-following, and coding tasks.

## Trade-offs

**Gain:** Higher effective throughput per forward pass than rejection-based SD, especially when draft-target diverge.  
**Lose:** Approximate (not exact) distribution — output distribution differs slightly from the target model's exact distribution. The error bound is a per-step approximation, not end-to-end output distribution guarantee.

## Relationship to Existing Work

- **vs CSD (arXiv 2604.13634):** CSD addresses false rejections via an online correction memory. SMC-SD addresses the same problem via a fundamentally different statistical framework (SMC vs. rejection sampling). Both are training-free.
- **vs P-EAGLE:** P-EAGLE improves draft generation speed; SMC-SD improves the verification/acceptance step.
- **vs SpecGuard:** SpecGuard targets reasoning step correctness; SMC-SD targets throughput.

## vLLM Integration

Not yet integrated. Implementation requires:
- Multi-particle draft generation (run draft model N times per step)
- Importance weight computation in the rejection sampler layer
- Resampling step replacement for the standard rejection sampler in `vllm/model_executor/layers/spec_decode/`

## Deferred Reason

Previously deferred from April 20-24 collect runs due to arXiv HTTP 403. Retrieved via web search snippet on April 24, 2026.
