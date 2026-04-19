---
title: "Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference"
source_type: arxiv-paper
source_url: https://arxiv.org/abs/2604.13634
collected: 2026-04-19
tags: [speculative-decoding, arxiv, acceptance-rate, latency, throughput, rejection-sampling]
---

# Calibrated Speculative Decoding (CSD)

**arXiv ID**: 2604.13634  
**Submitted**: April 15, 2026  
**Authors**: Xuwen Zhou, Fangxin Liu, Chao Wang, et al. (Shanghai Jiao Tong University; Alibaba Cloud Computing)

## Abstract / Core Problem

Standard speculative decoding suffers from **false rejections**: the draft model produces a semantically correct token but the rejection sampler discards it because the token differs lexically from what the target model's distribution expects. This wastes draft computation and reduces the effective acceptance rate below what it could be for semantically equivalent alternatives.

## Proposed Solution: CSD (Calibrated Speculative Decoding)

CSD is a **training-free** framework that retrofits existing draft-verify pipelines with two lightweight modules:

### 1. Online Correction Memory (OCM)

- During serving, maintains a rolling history of rejected (draft token, context) pairs
- Detects recurring divergence patterns: token pairs that are semantically equivalent but lexically different (e.g., "car" vs "automobile", different capitalization, minor phrasing variations)
- When the draft model proposes a token that was recently rejected in a similar context, OCM proposes **rescue candidates** — alternative tokens from the historical rejection pool that the target model is more likely to accept
- No model retraining: the memory is populated purely from live inference traffic

### 2. Semantic Consistency Gating (SCG)

- Verifies whether a rescue candidate from OCM is admissible using a **probability ratio test** rather than exact token matching
- Accepts a rescue token if its probability ratio (draft / target) falls within a calibrated threshold
- Prevents hallucinated or irrelevant rescues from being injected into the output stream
- Combines with the standard rejection sampler: the standard path runs first; OCM/SCG only activates on standard rejections

## Key Results

| Setup | Throughput speedup vs. baseline spec decode |
|-------|---------------------------------------------|
| Peak (best task/model) | **2.33×** |

- Evaluated across diverse LLMs (sizes not specified in search result summary)
- Outperforms existing speculative decoding methods on the measured benchmarks
- Training-free: plugs into any existing draft-verify pipeline without retraining either model

## Relationship to vLLM

From the paper: "CSD currently lacks integration with industrial-grade inference engines (e.g., vLLM), which is necessary to realize its full potential in large-scale, production-ready deployments."

This is a research paper; CSD is not currently integrated into vLLM. Implementation would require changes to vLLM's rejection sampler and the addition of a per-request (or global) correction memory store.

## Relationship to Existing Work

- Builds on standard speculative decoding (Leviathan et al., 2022)
- Distinguishes from EAGLE/P-EAGLE: those improve the draft model quality; CSD improves the verification step
- Distinguishes from calibrated/temperature-based approaches: CSD uses historical token-level statistics rather than scaling logits

## Limitations / Open Questions

- Memory overhead of OCM at scale: how large does the correction history need to be?
- Whether the rescue mechanism holds for structured outputs (code, JSON) where semantic equivalence is stricter
- Production integration path into vLLM's async rejection sampler
- Whether the 2.33× peak speedup holds at high concurrency (the P-EAGLE data showed spec decode advantages diminish at c=64)
